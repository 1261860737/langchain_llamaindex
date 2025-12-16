import os  
from pathlib import Path 
import json
from langchain_core.tools import Tool
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank

# 1. 定义一个加载 LlamaIndex 并转为 LangChain Tool 的函数
def get_llamaindex_tool(kb_name, kb_path):
    """
    加载 LlamaIndex 存储，并返回一个 LangChain Tool 对象
    """
    vs_path = Path(kb_path) / "vectorstore"
    config_path = Path(kb_path) / "config.json"
    

    # 如果这个文件不存在，说明该目录不是有效的 LlamaIndex 索引目录
    if not vs_path.exists() or not (vs_path / "docstore.json").exists():

        print(f"知识库 {kb_name} 尚未构建 LlamaIndex 索引，跳过加载。")
        return None
    # ---------------------------

    try:
        rerank_model_name = "None"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                rerank_model_name = config.get("rerank_model", "None")
        # A. 加载持久化的索引
        storage_context = StorageContext.from_defaults(persist_dir=str(vs_path))
        index = load_index_from_storage(storage_context)

        #  配置重排序处理器 (Node Postprocessor)
        node_postprocessors = []
        if rerank_model_name and rerank_model_name != "None":
            try:
                # 这是一个常用的开源重排序器封装
                reranker = SentenceTransformerRerank(
                    model=rerank_model_name, 
                    top_n=3 # 重排序后只取前 3 个最相关的给大模型
                )
                node_postprocessors.append(reranker)
                print(f"知识库 {kb_name} 已启用重排序: {rerank_model_name}")
            except Exception as e:
                print(f"重排序模型加载失败，将仅使用向量检索: {e}")
        
        # B. 转换为查询引擎 (Query Engine)
        query_engine = index.as_query_engine(
            similarity_top_k=10, 
            node_postprocessors=node_postprocessors
        )
        
        # C. 封装为 LangChain Tool
        def query_func(query: str) -> str:
            response = query_engine.query(query)
            return str(response)

        rag_tool = Tool(
            name=f"{kb_name}_knowledge_base_tool",
            func=query_func,
            description=f"用于查询关于 {kb_name} 的具体信息。输入应该是完整的问题。"
        )
        return rag_tool
        
    except Exception as e:
        print(f"加载知识库 {kb_name} 失败: {e}")
        return None