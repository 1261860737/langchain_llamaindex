import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import base64
from io import BytesIO
import streamlit as st
from llama_index.core import Settings


# 1. 平台列表
PLATFORMS = ["Ollama", "Xinference", "OpenAI", "ZhipuAI", "vLLM", "DashScope"] 
PlatformType = Literal["ZhipuAI", "DashScope"]

# 2. 获取 LLM 模型列表
def get_llm_models(platform_type: PlatformType, base_url: str="", api_key: str="EMPTY"):
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            client = ollama.Client(host=base_url)
            llm_models = [model["model"] for model in client.list()["models"] if "bert" not in model.details.families]
            return llm_models
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)
            llm_models = client.list_models()
            return [k for k,v in llm_models.items() if v.get("model_type") == "LLM"]
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    
    elif platform_type == "vLLM":   
        return [
            './models/Qwen/Qwen3-8B', # 替换为实际的vLLM模型路径
            'Qwen/Qwen2.5-72B-Instruct',
            'Qwen/Qwen2.5-7B-Instruct'
        ]
    
    elif platform_type == "DashScope":
        return [
            'qwen3-max', 
            'qwen-plus', 
            'qwen-turbo', 
            'qwen-long',
            'qwen2.5-72b-instruct',
            'qwen2.5-32b-instruct',
            'qwen2.5-7b-instruct',
        ]

    elif platform_type == "ZhipuAI":
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=api_key or os.getenv("ZHIPUAI_API_KEY"))
        return [
            'glm-4-alltools', 'glm-4-plus', 'glm-4-0520', 'glm-4',
            'glm-4-air', 'glm-4-airx', 'glm-4-long', 'glm-4-flashx', 'glm-4-flash'
        ]
    elif platform_type == "OpenAI":
        return [
            'gpt-4.1', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4o-mini', 'gpt-3.5-turbo',
            'qwen-max', 'deepseek-chat'
        ]
    

# 3. 获取 Embedding 模型名称列表 (复数)
def get_embedding_models(platform_type: PlatformType, base_url: str="", api_key: str="EMPTY"):
    
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            client = ollama.Client(host=base_url)
            embedding_models = [model["model"] for model in client.list()["models"] if "bert" in model.details.families]
            return embedding_models
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)
            embedding_models = client.list_models()
            return [k for k,v in embedding_models.items() if v.get("model_type") == "embedding"]
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []
    
    elif platform_type == "DashScope":
        return ['text-embedding-v1', 'text-embedding-v2', 'text-embedding-v3']

    elif platform_type == "OpenAI" or platform_type == "ZhipuAI":
        return ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
    
    return [] # 默认返回空列表


# 4. 获取 Chat Model 对象
def get_chatllm(
        platform_type: PlatformType,
        model: str,
        base_url: str = "",
        api_key: str = "",
        temperature: float = 0.1
):
    if platform_type == "Ollama":
        if not base_url:
            base_url = "http://127.0.0.1:11434"
        return ChatOllama(
            temperature=temperature,
            model=model,
            base_url=base_url
        )
    elif platform_type == "Xinference":
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        if not api_key:
            api_key = "EMPTY"
    elif platform_type == "ZhipuAI":
        if not base_url:
            base_url = "https://open.bigmodel.cn/api/paas/v4"
        if not api_key:
            api_key = os.getenv('ZHIPUAI_API_KEY')
    
    elif platform_type == "DashScope":
        if not base_url:
            base_url = os.getenv('DASHSCOPE_BASE_URL')
        if not api_key:
            api_key = os.getenv('DASHSCOPE_API_KEY')
            
    elif platform_type == "OpenAI":
        if not base_url:
            base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
            
    elif platform_type == "vLLM":
        if not base_url:
            base_url = os.getenv('vLLM_BASE_URL')
        if not api_key:     
            api_key = os.getenv('vLLM_API_KEY') or "EMPTY"

    return ChatOpenAI(
        temperature=temperature,
        model_name=model,
        streaming=True,
        base_url=base_url,
        api_key=api_key,
    )


# 5. --- 修复的关键：补充丢失的 get_embedding_model (单数) ---
# 这个函数用于返回 LangChain 的 Embeddings 对象，tools/naive_rag_tool.py 依赖它
def get_embedding_model(
        platform_type: PlatformType,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
):
    if platform_type == "Ollama":
        if not base_url:
            base_url = "http://127.0.0.1:11434"
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=base_url, model=model)
    
    elif platform_type == "Xinference":
        from langchain_community.embeddings.xinference import XinferenceEmbeddings
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        return XinferenceEmbeddings(server_url=base_url, model_uid=model)
    
    elif platform_type == "DashScope":
        from langchain_openai import OpenAIEmbeddings
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return OpenAIEmbeddings(
            base_url=base_url, 
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"), 
            model=model
        )
        
    else:
        # OpenAI, ZhipuAI, vLLM 通用逻辑
        from langchain_openai import OpenAIEmbeddings
        if not base_url:
            base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        return OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)


# 6. 配置 LlamaIndex 的全局 Settings
def config_llama_index(platform_type: str, embedding_model: str, llm_model: str = None, api_key: str = "", base_url: str = ""):
    try:
        if platform_type == "Ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            Settings.embed_model = OllamaEmbedding(
                model_name=embedding_model,
                base_url=base_url
            )
            
        elif platform_type == "OpenAI" or platform_type == "DashScope":
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            target_api_key = api_key
            target_base_url = base_url
            
            if platform_type == "OpenAI":
                target_api_key = target_api_key or os.getenv("OPENAI_API_KEY")
                target_base_url = target_base_url or os.getenv("OPENAI_BASE_URL")
            elif platform_type == "DashScope":
                target_api_key = target_api_key or os.getenv("DASHSCOPE_API_KEY")
                target_base_url = target_base_url or os.getenv("DASHSCOPE_BASE_URL")

            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=target_api_key,
                api_base=target_base_url
            )
            
        elif platform_type == "ZhipuAI":
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=api_key or os.getenv("ZHIPUAI_API_KEY"),
                api_base="https://open.bigmodel.cn/api/paas/v4"
            )
            
        elif platform_type == "Xinference":
            from llama_index.embeddings.openai import OpenAIEmbedding
            if not base_url:
                base_url = "http://127.0.0.1:9997/v1"
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_base=base_url,
                api_key="EMPTY"
            )
        
        return True
    except Exception as e:
        print(f"LlamaIndex 配置失败: {e}")
        return False


# 其他工具函数
def get_kb_names():
    kb_root = os.path.join(os.path.dirname(__file__), "kb")
    if not os.path.exists(kb_root):
        os.mkdir(kb_root)
    kb_names = [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]
    return kb_names

def get_img_base64(file_name: str) -> str:
    image_path = os.path.join(os.path.dirname(__file__), "img", file_name)
    with open(image_path, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base_str}"

def show_graph(graph):
    flow_state = StreamlitFlowState(
                       nodes=[StreamlitFlowNode(
                           id=node.id,
                           pos=(0,0),
                           data={"content": node.id},
                           node_type="input" if node.id == "__start__"
                                             else "output" if node.id == "__end__"
                                             else "default",
                       ) for node in graph.nodes.values()],
                       edges=[StreamlitFlowEdge(
                           id=str(enum),
                           source=edge.source,
                           target=edge.target,
                           animated=True,
                       ) for enum, edge in enumerate(graph.edges)],
                   )
    streamlit_flow('example_flow',
                   flow_state,
                   layout=TreeLayout(direction='down'), fit_view=True
    )

if __name__ == "__main__":
    print(get_embedding_models(platform_type="Ollama"))