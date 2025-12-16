import os
from typing import Literal, List
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
from langchain_openai import OpenAIEmbeddings 

# 尝试引入 LangchainEmbedding 适配器
# 智谱AI (ZhipuAI) 的 Embedding 也需要这个适配器才能被 LlamaIndex 使用
try:
    from llama_index.embeddings.langchain import LangchainEmbedding
except ImportError:
    LangchainEmbedding = None

# --- 自定义安全 Embedding 类 ---
# 用于解决部分厂商(如 ZhipuAI) API 对空字符串或特殊字符敏感的问题
class SafeOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts: List[str], chunk_size: int = None) -> List[List[float]]:
        # 将空字符串或纯空格替换为 "."，防止 API 报错 400 Bad Request
        cleaned_texts = [t if t and t.strip() else "." for t in texts]
        # 强制关闭本地 Token 长度检查，直接发送文本给 API
        return super().embed_documents(cleaned_texts, chunk_size)

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            text = "."
        return super().embed_query(text)
# --------------------------------------------------

PLATFORMS = ["Ollama", "Xinference", "OpenAI", "ZhipuAI", "vLLM", "DashScope"] 
PlatformType = Literal["ZhipuAI", "DashScope"]

# 1. 获取 LLM 模型列表 (保留 DashScope 用于对话)
def get_llm_models(platform_type: PlatformType, base_url: str="", api_key: str="EMPTY"):
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url: base_url = "http://127.0.0.1:11434"
            client = ollama.Client(host=base_url)
            return [model["model"] for model in client.list()["models"] if "bert" not in model.details.families]
        except Exception: return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url: base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)
            return [k for k,v in client.list_models().items() if v.get("model_type") == "LLM"]
        except Exception: return []
    elif platform_type == "vLLM":   
        return ['./models/Qwen/Qwen3-8B', 'Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']
    
    # ✅ DashScope: 保留用于对话 (Chat)
    elif platform_type == "DashScope":
        return ['qwen3-max', 'qwen-plus', 'qwen-turbo', 'qwen-long', 'qwen2.5-72b-instruct', 'qwen2.5-32b-instruct']
    
    elif platform_type == "ZhipuAI":
        return ['glm-4-alltools', 'glm-4-plus', 'glm-4-0520', 'glm-4', 'glm-4-air', 'glm-4-airx', 'glm-4-flash']
    elif platform_type == "OpenAI":
        return ['gpt-4.1', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4o-mini', 'gpt-3.5-turbo', 'deepseek-chat']
    return []

# 2. 获取 Embedding 模型列表 (移除 DashScope)
def get_embedding_models(platform_type: PlatformType, base_url: str="", api_key: str="EMPTY"):
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url: base_url = "http://127.0.0.1:11434"
            client = ollama.Client(host=base_url)
            return [model["model"] for model in client.list()["models"] if "bert" in model.details.families]
        except Exception: return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url: base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)
            return [k for k,v in client.list_models().items() if v.get("model_type") == "embedding"]
        except Exception: return []
    
    # ❌ DashScope: 返回空列表，不再支持构建知识库
    elif platform_type == "DashScope":
        return [] 

    # ✅ ZhipuAI: 返回正确的模型名称
    elif platform_type == "ZhipuAI":
        return ['embedding-2', 'embedding-3'] 
    
    elif platform_type == "OpenAI":
        return ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
    
    return []

# 3. 获取 Chat Model 对象
def get_chatllm(platform_type: PlatformType, model: str, base_url: str = "", api_key: str = "", temperature: float = 0.1):
    if platform_type == "Ollama":
        if not base_url: base_url = "http://127.0.0.1:11434"
        return ChatOllama(temperature=temperature, model=model, base_url=base_url)
    elif platform_type == "Xinference":
        if not base_url: base_url = "http://127.0.0.1:9997/v1"
        if not api_key: api_key = "EMPTY"
    elif platform_type == "ZhipuAI":
        if not base_url: base_url = "https://open.bigmodel.cn/api/paas/v4"
        if not api_key: api_key = os.getenv('ZHIPUAI_API_KEY')
    elif platform_type == "DashScope":
        if not base_url: base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not api_key: api_key = os.getenv('DASHSCOPE_API_KEY')
    elif platform_type == "OpenAI":
        if not base_url: base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key: api_key = os.getenv('OPENAI_API_KEY')
    elif platform_type == "vLLM":
        if not base_url: base_url = os.getenv('vLLM_BASE_URL')
        if not api_key: api_key = os.getenv('vLLM_API_KEY') or "EMPTY"
    return ChatOpenAI(temperature=temperature, model_name=model, streaming=True, base_url=base_url, api_key=api_key, request_timeout=120, max_retries=1)

# 4. 获取 LangChain Embedding 对象
def get_embedding_model(platform_type: PlatformType, model: str = "", base_url: str = "", api_key: str = ""):
    if platform_type == "Ollama":
        if not base_url: base_url = "http://127.0.0.1:11434"
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=base_url, model=model)
    elif platform_type == "Xinference":
        from langchain_community.embeddings.xinference import XinferenceEmbeddings
        if not base_url: base_url = "http://127.0.0.1:9997/v1"
        return XinferenceEmbeddings(server_url=base_url, model_uid=model)
    
    # ✅ ZhipuAI: 使用 SafeOpenAIEmbeddings 以获得更好的稳定性
    elif platform_type == "ZhipuAI":
        if not base_url: base_url = "https://open.bigmodel.cn/api/paas/v4"
        return SafeOpenAIEmbeddings(
            base_url=base_url, 
            api_key=api_key or os.getenv("ZHIPUAI_API_KEY"), 
            model=model,
            check_embedding_ctx_length=False # 禁用本地 Token 检查
        )
        
    else:
        # OpenAI 或其他兼容接口
        from langchain_openai import OpenAIEmbeddings
        if not base_url: base_url = os.getenv('OPENAI_BASE_URL')
        if not api_key: api_key = os.getenv('OPENAI_API_KEY')
        return OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)

# 5. 配置 LlamaIndex 的全局 Settings (逻辑简化)
def config_llama_index(platform_type: str, embedding_model: str, llm_model: str = None, api_key: str = "", base_url: str = ""):
    try:
        # 智能检测：ZhipuAI 走 SafeAdapter
        use_safe_adapter = False
        target_base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        
        if platform_type == "ZhipuAI" or "open.bigmodel.cn" in target_base_url:
            use_safe_adapter = True

        if use_safe_adapter:
            if LangchainEmbedding is None: raise ImportError("请安装 llama-index-embeddings-langchain")
            print(f"🔧 检测到 {platform_type} 环境，正在应用安全适配器...")
            lc_embed_model = get_embedding_model(
                platform_type=platform_type, 
                model=embedding_model, 
                base_url=target_base_url, 
                api_key=api_key
            )
            Settings.embed_model = LangchainEmbedding(lc_embed_model)

        elif platform_type == "Ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding
            if not base_url: base_url = "http://127.0.0.1:11434"
            Settings.embed_model = OllamaEmbedding(model_name=embedding_model, base_url=base_url)
            
        elif platform_type == "OpenAI":
            from llama_index.embeddings.openai import OpenAIEmbedding
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                api_base=base_url or os.getenv("OPENAI_BASE_URL")
            )
            
        elif platform_type == "Xinference":
             from llama_index.embeddings.openai import OpenAIEmbedding
             if not base_url: base_url = "http://127.0.0.1:9997/v1"
             Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_base=base_url, api_key="EMPTY")
        
        return True
    except Exception as e:
        print(f"LlamaIndex 配置失败: {e}")
        if 'streamlit' in str(type(st)):
            st.error(f"LlamaIndex 配置失败: {e}")
        return False

# 其他工具函数保持不变
def get_kb_names():
    kb_root = os.path.join(os.path.dirname(__file__), "kb")
    if not os.path.exists(kb_root): os.mkdir(kb_root)
    return [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]

def get_img_base64(file_name: str) -> str:
    image_path = os.path.join(os.path.dirname(__file__), "img", file_name)
    if not os.path.exists(image_path): return ""
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

def show_graph(graph):
    try:
        flow_state = StreamlitFlowState(
            nodes=[StreamlitFlowNode(id=node.id, pos=(0,0), data={"content": node.id}, node_type="input" if node.id == "__start__" else "output" if node.id == "__end__" else "default") for node in graph.nodes.values()],
            edges=[StreamlitFlowEdge(id=str(enum), source=edge.source, target=edge.target, animated=True) for enum, edge in enumerate(graph.edges)],
        )
        streamlit_flow('example_flow', flow_state, layout=TreeLayout(direction='down'), fit_view=True)
    except Exception: pass

if __name__ == "__main__":
    pass