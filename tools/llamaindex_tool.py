import os
import json
from pathlib import Path
from langchain_core.tools import Tool
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# 引入 Query 泛化所需的 LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入 OpenAILike
try:
    from llama_index.llms.openai_like import OpenAILike
except ImportError:
    OpenAILike = None

from utils import get_embedding_model
try:
    from llama_index.embeddings.langchain import LangchainEmbedding
except ImportError:
    LangchainEmbedding = None

# --- ✅ 新增：Query 泛化函数 ---
def stepback_prompting_expansion(query: str, api_key: str = None) -> str:
    """利用 LLM 将具体问题泛化为通用问题"""
    try:
        if not api_key: return query
        
        examples = [
            {"input": "这篇关于Transformer的论文是如何解决长文本效率低下的？", "output": "LLM处理超长文本时的主要挑战和架构改进方案有哪些？"},
            {"input": "指令微调对提升模型在数学任务上的表现有帮助吗？", "output": "指令微调在提升LLM特定任务能力方面扮演了什么角色？"},
        ]
        example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
        few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个文献检索专家，请将用户的具体问题泛化为更适合检索的通用问题。仅返回泛化后的问题。"),
            few_shot_prompt,
            ("user", "{question}"),
        ])
        
        # 使用 DashScope 进行泛化 (因为用户现在主要用这个)
        llm = ChatOpenAI(
            model="qwen-plus", 
            temperature=0.1, 
            openai_api_key=api_key, 
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        chain = prompt | llm | StrOutputParser()
        expanded_query = chain.invoke({"question": query})
        print(f"🔄 Query 泛化: {query} -> {expanded_query}")
        return expanded_query
    except Exception as e:
        print(f"⚠️ 泛化失败: {e}")
        return query
# --------------------------------

def get_llamaindex_tool(kb_name, kb_path):
    vs_path = Path(kb_path) / "vectorstore"
    config_path = Path(kb_path) / "config.json"
    
    if not vs_path.exists() or not (vs_path / "docstore.json").exists():
        print(f"知识库 {kb_name} 尚未构建 LlamaIndex 索引，跳过加载。")
        return None

    try:
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        
        kb_platform = config.get("platform", "OpenAI")
        kb_embed_model_name = config.get("embedding_model", "text-embedding-3-small")
        rerank_model_name = config.get("rerank_model", "None")

        # 1. 配置 Embedding
        embed_model = None
        if LangchainEmbedding:
            lc_embed = get_embedding_model(platform_type=kb_platform, model=kb_embed_model_name)
            embed_model = LangchainEmbedding(lc_embed)

        # 2. 配置 LLM (使用 OpenAILike 绕过验证)
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if OpenAILike and api_key:
            llm = OpenAILike(
                model="qwen-plus",
                api_key=api_key,
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                is_chat_model=True,
                temperature=0.1,
                timeout=120.0,
                max_retries=2
            )
            Settings.llm = llm

        # 3. 加载索引
        storage_context = StorageContext.from_defaults(persist_dir=str(vs_path))
        index = load_index_from_storage(storage_context, embed_model=embed_model)

        # 4. 混合检索
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=6)
        bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=6)
        
        from llama_index.core.retrievers import BaseRetriever
        class HybridRetriever(BaseRetriever):
            def __init__(self, vector, bm25):
                self.vector = vector
                self.bm25 = bm25
                super().__init__()
            def _retrieve(self, query_bundle):
                v_nodes = self.vector.retrieve(query_bundle)
                b_nodes = self.bm25.retrieve(query_bundle)
                all_nodes = {n.node.node_id: n for n in v_nodes}
                for n in b_nodes:
                    if n.node.node_id not in all_nodes:
                        all_nodes[n.node.node_id] = n
                return list(all_nodes.values())

        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

        # 5. 重排序
        node_postprocessors = []
        if rerank_model_name and rerank_model_name != "None":
            try:
                reranker = SentenceTransformerRerank(model=rerank_model_name, top_n=3)
                node_postprocessors.append(reranker)
            except Exception: pass

        # 6. 查询引擎
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=node_postprocessors,
            llm=llm
        )
        
        def query_func(query: str) -> str:
            # --- ✅ 恢复调用泛化逻辑 ---
            # 只有当 query 比较短或者意图不明确时才泛化，这里简单全部尝试
            final_query = stepback_prompting_expansion(query, api_key=api_key)
            response = query_engine.query(final_query)
            return str(response)

        return Tool(
            name=f"{kb_name}_knowledge_base_tool",
            func=query_func,
            description=f"用于查询关于 {kb_name} 的信息。"
        )
        
    except Exception as e:
        print(f"加载知识库 {kb_name} 失败: {e}")
        return None