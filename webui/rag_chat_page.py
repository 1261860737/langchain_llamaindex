import streamlit as st
import uuid
import os
from pathlib import Path

from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from tools.llamaindex_tool import get_llamaindex_tool

RAG_PAGE_INTRODUCTION = "你好，我是智能助手，当前页面为`RAG 对话模式`，可以在对话让大模型基于左侧所选知识库进行回答，有什么可以帮助你的吗？"

# 使用 st.cache_resource 避免每次刷新页面都重建索引
@st.cache_resource(show_spinner=False)
def load_kb_tool(kb_name, kb_path):
    return get_llamaindex_tool(kb_name, kb_path)

# Graph Response 处理逻辑
def graph_response(graph, input):
    for event in graph.stream(
        {"messages": input},
        config={"configurable": {"thread_id": str(uuid.uuid4())}}, 
        stream_mode="messages",
    ):
        if type(event[0]) == AIMessageChunk:
            if len(event[0].tool_calls):
                st.session_state["rag_tool_calls"].append(
                    {
                        "status": "正在查询...",
                        "knowledge_base": event[0].tool_calls[0]["name"].replace("_knowledge_base_tool", ""),
                        "query": str(event[0].tool_calls[0]["args"].get("query", "无查询参数")),
                    }
                )
            yield event[0].content
        elif type(event[0]) == ToolMessage:
            if len(st.session_state["rag_tool_calls"]) > 0:
                st.session_state["rag_tool_calls"][-1]["status"] = "已完成工具调用"
                st.session_state["rag_tool_calls"][-1]["content"] = event[0].content

# 获取回答的主逻辑
def get_rag_chat_response(platform, model, temperature, messages, selected_kbs, KBS, base_url, api_key):
    if not selected_kbs:
        yield "请先在侧边栏选择至少一个知识库。"
        return

    tools = [KBS[k] for k in selected_kbs if k in KBS]
    
    if not tools:
        yield "选中的知识库未成功加载，请检查知识库状态。"
        return

    tool_node = ToolNode(tools=tools)

    def call_model(state):
        llm = get_chatllm(platform, model, base_url=base_url, api_key=api_key, temperature=temperature)
        llm_with_tools = llm.bind_tools(tools)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    
    graph = workflow.compile()
    yield from graph_response(graph, messages)

# 修复 display_chat_history 报错
def display_chat_history():
    for message in st.session_state["rag_chat_history_with_tool_call"]:
        with st.chat_message(message["role"], avatar=get_img_base64("robot.png") if message["role"] == "assistant" else None):
            if "tool_calls" in message.keys():
                for tool_call in message["tool_calls"]:
                    status = tool_call.get("status", "已完成工具调用")
                    with st.status(status, expanded=False):
                        st.write("已调用知识库: ", tool_call.get("knowledge_base", "未知"))
                        if "query" in tool_call:
                            st.write("查询语句:")
                            st.code(tool_call.get("query", ""), wrap_lines=True)
                        st.write("知识库检索结果：")
                        content = tool_call.get("content")
                        if content:
                            st.write(content)
                        else:
                            st.warning("⚠️ 工具调用中断或未返回结果")
            st.write(message.get("content", ""))

# 主页面函数
def rag_chat_page():
    kbs = get_kb_names()
    KBS = dict()
    kb_root = Path(__file__).resolve().parents[1] / "kb"
    
    for k in kbs:
        kb_path = kb_root / k
        tool = load_kb_tool(k, kb_path)
        if tool:
            KBS[k] = tool

    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]
    if "rag_chat_history_with_tool_call" not in st.session_state:
        st.session_state["rag_chat_history_with_tool_call"] = [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]
    if "rag_tool_calls" not in st.session_state:
        st.session_state["rag_tool_calls"] = []

    # --- 侧边栏配置 (将配置和清空移到这里) ---
    with st.sidebar:
        st.subheader("🤖 模型配置")
        platform = st.selectbox("模型平台", PLATFORMS)
        llm_models = get_llm_models(platform, st.session_state.get("base_url", ""), st.session_state.get("api_key", ""))
        if not llm_models: llm_models = ["加载失败或列表为空"]
        model = st.selectbox("选择模型", llm_models)
        temperature = st.slider("Temperature", 0.1, 1., 0.1)
        history_len = st.slider("历史消息长度", 1, 10, 5)
        
        st.divider()
        st.subheader("📚 知识库选择")
        selected_kbs = st.sidebar.multiselect("选择知识库", kbs)
        
        st.divider()
        st.subheader("🔑 密钥配置")
        base_url = st.text_input("Base URL", help="如 Ollama 或 DashScope 的 base_url", key="base_url")
        api_key = st.text_input("API Key", help="API Key", type="password", key="api_key")
        
        st.divider()
        # 清空按钮放这里
        st.button("🗑️ 清空对话", on_click=lambda: st.session_state.update({
            "rag_chat_history": [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}],
            "rag_chat_history_with_tool_call": [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}],
            "rag_tool_calls": []
        }), use_container_width=True)

    # 主区域显示历史
    display_chat_history()

    # --- ✅ 修复点：chat_input 移出 columns，独占主层级 ---
    # 这样它就会自动吸附在页面最底部
    input = st.chat_input("请输入您的问题")
    # --------------------------------------------------
    
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["rag_chat_history"] += [{"role": 'user', "content": input}]
        st.session_state["rag_chat_history_with_tool_call"] += [{"role": 'user', "content": input}]

        stream_response = get_rag_chat_response(
            platform, model, temperature,
            st.session_state["rag_chat_history"][-history_len:],
            selected_kbs, KBS,
            st.session_state.get("base_url", ""), st.session_state.get("api_key", "")
        )

        with st.chat_message("assistant", avatar=get_img_base64("robot.png")):
            response = st.write_stream(stream_response)
        
        st.session_state["rag_chat_history"] += [{"role": 'assistant', "content": response}]
        st.session_state["rag_chat_history_with_tool_call"] += [{
            "role": 'assistant', "content": response, "tool_calls": st.session_state["rag_tool_calls"]
        }]
        st.session_state["rag_tool_calls"] = []