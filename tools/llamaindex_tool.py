import os
import json
from pathlib import Path
from langchain_core.tools import Tool
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# --- ✅ 关键修改：引入 OpenAILike 代替 OpenAI ---
# OpenAILike 不会校验模型名称，适合 DashScope/DeepSeek 等兼容接口
try:
    from llama_index.llms.openai_like import OpenAILike
except ImportError:
    # 如果没安装，尝试回退或提示（通常 llama-index-llms-openai-like 包是自带的或需单独安）
    OpenAILike = None

# 引入 utils 中的函数来构建 embedding model
from utils import get_embedding_model
try:
    from llama_index.embeddings.langchain import LangchainEmbedding
except ImportError:
    LangchainEmbedding = None

def get_llamaindex_tool(kb_name, kb_path):
    vs_path = Path(kb_path) / "vectorstore"
    config_path = Path(kb_path) / "config.json"
    
    if not vs_path.exists() or not (vs_path / "docstore.json").exists():
        print(f"知识库 {kb_name} 尚未构建 LlamaIndex 索引，跳过加载。")
        return None

    try:
        # 1. 读取配置
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        
        kb_platform = config.get("platform", "OpenAI")
        kb_embed_model_name = config.get("embedding_model", "text-embedding-3-small")
        rerank_model_name = config.get("rerank_model", "None")

        # 2. 配置 Embedding 模型
        embed_model = None
        if LangchainEmbedding:
            lc_embed = get_embedding_model(
                platform_type=kb_platform,
                model=kb_embed_model_name
            )
            embed_model = LangchainEmbedding(lc_embed)

        # --- ✅ 新增：配置 OpenAILike LLM (绕过模型名验证) ---
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_BASE_URL")
        if not api_key:
            print("⚠️ 未检测到 DASHSCOPE_API_KEY，LlamaIndex 可能无法生成回答。")
        
        if OpenAILike:
            llm = OpenAILike(
                model="qwen-plus",  # 现在可以随意传模型名了
                api_key=api_key,
                api_base=base_url,
                is_chat_model=True, # 声明这是对话模型
                temperature=0.1,
                timeout=120.0,      # 超时时间
                max_retries=2,
                reuse_client=False
            )
            # 将配置应用到全局
            Settings.llm = llm
        else:
            print("❌ 缺少 llama-index-llms-openai-like 库，请运行 `pip install llama-index-llms-openai-like`")
        # -----------------------------------------------------------

        # 3. 加载索引
        storage_context = StorageContext.from_defaults(persist_dir=str(vs_path))
        index = load_index_from_storage(
            storage_context, 
            embed_model=embed_model 
        )

        # 4. 构建混合检索
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
            except Exception as e:
                print(f"重排序模型加载失败: {e}")

        # 6. 转换为 Query Engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=node_postprocessors,
            llm=llm  # 显式传入 OpenAILike LLM
        )
        
        def query_func(query: str) -> str:
            response = query_engine.query(query)
            return str(response)

        return Tool(
            name=f"{kb_name}_knowledge_base_tool",
            func=query_func,
            description=f"用于查询关于 {kb_name} 的信息。"
        )
        
    except Exception as e:
        print(f"加载知识库 {kb_name} 失败: {e}")
        return None