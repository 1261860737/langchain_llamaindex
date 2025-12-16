import os
import json
import streamlit as st
from pathlib import Path
import shutil

# 引入 utils 中的配置函数
from utils import PLATFORMS, get_embedding_models, get_kb_names, config_llama_index

# --- 引入 LlamaIndex 核心组件 ---
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)

# 1. 保存配置
def save_config_to_json(kb_path, platform, embedding_model, rerank_model):
    config_path = kb_path / "config.json"
    config_data = {
        "platform": platform,
        "embedding_model": embedding_model,
        "rerank_model": rerank_mode
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f)

# 2. 加载配置
def load_config_from_json(kb_path):
    config_path = kb_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None

# 3. 简单的文件查重
def check_duplicate_file(file_name, file_storage_path):
    return (file_storage_path / file_name).exists()

# 4. 核心：使用 LlamaIndex 构建/更新索引
def build_knowledge_base(kb_path, file_storage_path, platform, embedding_model):
    """
    使用 LlamaIndex 读取 file_storage_path 下的所有文件并构建索引
    """
    # A. 配置全局 Settings (关键步骤)
    # 这里调用我们在 utils.py 中新写的函数
    success = config_llama_index(platform, embedding_model)
    if not success:
        raise Exception("LlamaIndex 模型配置失败，请检查 utils.py 或环境配置")

    try:
        # B. 使用 SimpleDirectoryReader 自动加载目录下所有文件
        # recursive=True 允许读取子目录（虽然目前逻辑只存根目录）
        reader = SimpleDirectoryReader(
            input_dir=str(file_storage_path),
            recursive=True,
            filename_as_id=True # 使用文件名作为 ID，有助于去重
        )
        documents = reader.load_data()
        
        if not documents:
            return "⚠️ 目录下没有可读取的文件，无法构建索引。"

        # C. 构建向量索引 (自动切分 + 向量化)
        # show_progress=True 会在终端显示进度条
        index = VectorStoreIndex.from_documents(documents, show_progress=True)

        # D. 持久化存储索引
        # 我们把索引存到 kb_name/vectorstore 目录下
        vs_path = kb_path / "vectorstore"
        index.storage_context.persist(persist_dir=str(vs_path))
        
        return f"✅ 索引构建成功！LlamaIndex 共处理了 {len(documents)} 个文档片段。"
        
    except Exception as e:
        raise e

# 界面主逻辑
def knowledge_base_page():
    if "selected_kb" not in st.session_state:
        st.session_state["selected_kb"] = ""
    
    st.title("知识库管理 (LlamaIndex Powered)")
    
    kb_names = get_kb_names()
    selected_kb = st.selectbox("请选择知识库", ["新建知识库"] + kb_names)

    # --- 场景 1: 新建知识库 ---
    if selected_kb == "新建知识库":
        status_placeholder = st.empty()
        with status_placeholder.status("知识库配置", expanded=True) as s:
            cols = st.columns(2)
            kb_name = cols[0].text_input("请输入知识库名称", placeholder="请使用英文，如：companies_information")
            # LlamaIndex 默认用本地文件存储，这里暂时不需要选向量库类型，或者作为一个占位符
            vs_type = cols[1].selectbox("向量库存储方式", ["Local (Default)"], disabled=True)
            
            st.text_area("请输入知识库描述", placeholder="如：介绍企业基本信息")
            
            cols = st.columns(2)
            platform = cols[0].selectbox("请选择 Embedding 平台", PLATFORMS)
            embedding_models = get_embedding_models(platform)
            embedding_model = cols[1].selectbox("请选择 Embedding 模型", embedding_models)
            rerank_options = ["None", "BAAI/bge-reranker-base", "BAAI/bge-reranker-large"] 
            rerank_model = st.selectbox("请选择重排序模型 (Rerank)", rerank_options)
            
            submit = st.button("创建知识库")
            
            if submit:
                if not kb_name.strip():
                    st.error("知识库名称不能为空")
                    s.update(state="error")
                    st.stop()
                
                kb_root = Path(__file__).resolve().parents[1] / "kb"
                kb_path = kb_root / kb_name
                file_storage_path = kb_path / "files"
                vs_path = kb_path / "vectorstore"
                
                if kb_path.exists():
                    st.error("知识库已存在")
                    s.update(state="error")
                    st.stop()
                
                # 创建目录结构
                kb_path.mkdir(parents=True)
                file_storage_path.mkdir()
                vs_path.mkdir()
                
                # 保存配置
                save_config_to_json(kb_path, platform, embedding_model, rerank_model)

                st.success(f"创建知识库 {kb_name} 成功")
                s.update(label=f'已创建知识库"{kb_name}"', expanded=False, state="complete")
                
                # 刷新页面状态
                st.session_state["selected_kb"] = kb_name
                st.rerun()

    # --- 场景 2: 管理已有知识库 ---
    else:
        kb_root = Path(__file__).resolve().parents[1] / "kb"
        kb_path = kb_root / selected_kb
        file_storage_path = kb_path / "files"
        vs_path = kb_path / "vectorstore"
        
        # 加载配置
        config = load_config_from_json(kb_path)
        if config:
            platform = config.get("platform", "OpenAI")
            embedding_model = config.get("embedding_model", "text-embedding-3-small")
            current_rerank = config.get("rerank_model", "None")
        else:
            st.warning("未找到配置文件，部分功能可能受限。")
            platform = "OpenAI"
            embedding_model = "text-embedding-3-small"
            current_rerank = "None"

        st.info(f"当前配置: 平台 `{platform}` | 模型 `{embedding_model}` | Rerank模型 `{current_rerank}`")

        # 这里的 uploader_placeholder 用于容纳上传控件
        uploader_placeholder = st.empty()
        
        with uploader_placeholder.container():
            files = st.file_uploader(
                "上传文件 (支持 PDF, Word, MD, TXT 等)", 
                type=["md", "txt", "pdf", "docx"], 
                accept_multiple_files=True
            )
            
            # 使用 form 或者 button 来触发处理
            if st.button("上传并重建索引") and files:
                
                progress_bar = st.progress(0, text="准备处理...")
                
                # 1. 保存文件到磁盘
                new_files_count = 0
                for file in files:
                    file_path = file_storage_path / file.name
                    if not check_duplicate_file(file.name, file_storage_path):
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                        new_files_count += 1
                    else:
                        # 如果文件已存在，我们选择覆盖它，以确保内容最新
                        # 或者跳过。这里选择覆盖，因为用户显式上传了。
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                
                progress_bar.progress(30, text=f"文件保存完成，开始构建索引...")
                
                # 2. 调用 LlamaIndex 全自动处理
                # 注意：我们重新读取整个文件夹来构建索引，这是最简单且一致性最好的方式
                try:
                    msg = build_knowledge_base(
                        kb_path, 
                        file_storage_path, 
                        platform, 
                        embedding_model
                    )
                    progress_bar.progress(100, text="完成！")
                    st.success(msg)
                    st.balloons()
                except Exception as e:
                    st.error(f"构建失败: {str(e)}")
                    progress_bar.empty()

        # 展示当前已有的文件列表（可选）
        with st.expander("查看已上传文件"):
            if file_storage_path.exists():
                existing_files = os.listdir(file_storage_path)
                if existing_files:
                    st.write(existing_files)
                else:
                    st.write("暂无文件")