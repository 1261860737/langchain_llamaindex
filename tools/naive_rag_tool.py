import os
import logging
from typing import Optional, Dict, Any, Literal
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from utils import get_embedding_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pickle
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义支持的重排序类型
RERANK_TYPES = Literal["flashrank", "cohere", "jina"]

class RAGConfig:
    def __init__(self, 
                 k: int = 3,
                 score_threshold: float = 0.15,
                 model_name: str = "gpt-4o-mini",
                 persist_directory: Optional[str] = None,
                 rerank_type: RERANK_TYPES = "jina",
                 rerank_api_key: Optional[str] ="",
                 ollama_base_url: str = "http://localhost:11434"):
        self.k = k
        self.score_threshold = score_threshold
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.rerank_type = rerank_type
        self.rerank_api_key = rerank_api_key
        self.ollama_base_url = ollama_base_url

def get_reranker(config: RAGConfig):
    """获取重排序器"""
    try:
        if config.rerank_type == "flashrank":
            return FlashrankRerank()
        
        elif config.rerank_type == "cohere":
            from langchain_community.document_compressors import CohereRerank
            if not config.rerank_api_key:
                raise ValueError("Cohere rerank requires API key")
            return CohereRerank(
                api_key=config.rerank_api_key,
                model="rerank-multilingual-v2.0"  # 或 "rerank-multilingual-v2.0"
            )
        
        elif config.rerank_type == "jina":
            from langchain_community.document_compressors import JinaRerank
            if not config.rerank_api_key:
                config.rerank_api_key = os.getenv("JINA_API_KEY")
            return JinaRerank(
                jina_api_key=config.rerank_api_key,
                model="jina-reranker-v2-base-multilingual" # 或 "jina-rerank-v1-base-zh"
            )
        
        else:
            raise ValueError(f"Unsupported rerank type: {config.rerank_type}")
            
    except Exception as e:
        logger.warning(f"重排序器初始化失败: {str(e)}")
        return None

# 加载本地文档用于 BM25 检索
def load_documents(pkl_path: str) -> list:
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"文档文件不存在: {pkl_path}")
        raise
    except Exception as e:
        logger.error(f"加载文档时发生错误: {str(e)}")
        raise

# 主函数：构建混合 + rerank + 泛化支持的 Rag 工具
def get_advanced_rag_tool(vectorstore_name: str, 
                         document_pkl_path: str, 
                         api_key: str,
                         config: Optional[RAGConfig] = None) -> Any:
    try:
        from langchain_chroma import Chroma

        if config is None:
            config = RAGConfig()

        # 设置持久化目录
        persist_dir = config.persist_directory or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "kb", 
            vectorstore_name, 
            "vectorstore"
        )

        # 将数据添加到向量数据库中
        vectorstore = Chroma(
            collection_name=vectorstore_name,
            embedding_function=get_embedding_model(platform_type="OpenAI", model="text-embedding-3-small"),
            # embedding_function=get_embedding_model(platform_type="Ollama", model="bge-m3:latest"),
            persist_directory=persist_dir,
        )

        chroma_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.k,
                "score_threshold": config.score_threshold,
            }
        )

        # BM25 文本检索器
        documents = load_documents(document_pkl_path)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = config.k

        # 混合检索器
        ensemble_retriever = EnsembleRetriever(retrievers=[chroma_retriever, bm25_retriever])

        # 添加重排序
        try:
            reranker = get_reranker(config)
            if reranker:
                rerank_retriever = ContextualCompressionRetriever(
                    base_compressor=reranker,
                    base_retriever=ensemble_retriever,
                )
            else:
                rerank_retriever = ensemble_retriever
        except Exception as e:
            logger.warning(f"重排序初始化失败，使用基础检索器: {str(e)}")
            rerank_retriever = ensemble_retriever

        # 构造工具
        retriever_tool = create_retriever_tool(
            rerank_retriever,
            name=f"{vectorstore_name}_knowledge_tool",
            description=f"Search and retrieve information from {vectorstore_name} with hybrid + rerank + stepback.",
        )
        retriever_tool.response_format = "content"

        # query 泛化 + 检索调用函数
        def retrieval_func(query: str) -> Dict[str, str]:
            try:
                general_query = stepback_prompting_expansion(query, api_key, config.model_name)
            except Exception as e:
                logger.warning(f"Query泛化失败，使用原始query: {str(e)}")
                general_query = query

            try:
                docs = rerank_retriever.invoke(general_query)
                return {
                    f"已知内容 {i+1}": doc.page_content.replace(doc.metadata.get("source", "") + "\n\n", "")
                    for i, doc in enumerate(docs)
                }
            except Exception as e:
                logger.error(f"检索过程发生错误: {str(e)}")
                return {"error": "检索失败，请稍后重试"}

        retriever_tool.func = retrieval_func
        return retriever_tool

    except Exception as e:
        logger.error(f"创建RAG工具时发生错误: {str(e)}")
        raise

# query 泛化函数
def stepback_prompting_expansion(query: str, api_key: str, model_name: str = "gpt-4") -> str:
    """
    将用户提出的AI深度学习技术问题泛化为通用文献检索问题。
    如果泛化失败，则自动回退到原始query。
    """
    try:
        examples = [
            {
                "input": "这篇关于改进Transformer架构的论文是如何解决LLM处理超长文本时效率低下的问题的？",
                "output": "在处理超长文本序列时，当前LLM面临哪些主要挑战？有哪些代表性的架构改进方案？"
            },
            {
                "input": "论文中提出的稀疏注意力机制是否显著提升了LLM在长上下文任务上的推理速度？",
                "output": "在提升LLM的长上下文处理效率方面，稀疏注意力机制发挥了哪些关键作用？"
            },
            {
                "input": "这篇使用Mixture-of-Experts架构的LLM相比标准的Transformer架构在训练成本和性能上有哪些优势？",
                "output": "在构建大型语言模型时，稀疏专家混合架构相较稠密Transformer架构有哪些主要优势？"
            },
            {
                "input": "指令微调是否已经成为提升LLM下游任务通用能力的必要步骤？",
                "output": "在提升LLM的通用任务执行能力方面，指令微调扮演了何种关键角色？"
            },
            {
                "input": "上下文学习能力是否让LLM在小样本场景下超越了需要大量标注数据的传统监督学习模型？",
                "output": "在小样本学习任务中，LLM的上下文学习能力相较传统监督学习方法有何优势？"
            },
        ]

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个深度学习领域的文献检索专家，擅长将非常具体的AI论文问题泛化为更通用的检索问题，以帮助在文献、教材中查找答案。参考以下例子："),
            few_shot_prompt,
            ("user", "{question}"),
        ])

        llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=api_key)
        chain = prompt | llm | StrOutputParser()

        expanded_query = chain.invoke({"question": query})
        if not expanded_query.strip():
            logger.warning("LLM返回空字符串，使用原始query")
            return query
        return expanded_query

    except Exception as e:
        logger.error(f"Query泛化失败: {str(e)}")
        return query




