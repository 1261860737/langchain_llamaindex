import streamlit as st
import json
import os
import uuid
from pathlib import Path 

from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from tools.llamaindex_tool import get_llamaindex_tool

RAG_PAGE_INTRODUCTION = "你好，我是智能助手，当前页面为`RAG 对话模式`，可以在对话让大模型基于左侧所选知识库进行回答，有什么可以帮助你的吗？"

# --- ✅ 新增：定义一个带缓存的加载函数 ---
@st.cache_resource(show_spinner=False)
def load_kb_tool(kb_name, kb_path):
    return get_llamaindex_tool(kb_name, kb_path)

# Graph Response 处理逻辑保持不变
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
            status_placeholder = st.empty()
            with (status_placeholder.status("正在查询...", expanded=True) as s):
                st.write("已调用 `", event[0].name.replace("_knowledge_base_tool", ""), "` 知识库进行查询")
                continue_save = False
                if len(st.session_state["rag_tool_calls"]):
                    if "content" not in st.session_state["rag_tool_calls"][-1].keys() \
                            and event[0].name.replace("_knowledge_base_tool", "") == \
                        st.session_state["rag_tool_calls"][-1]["knowledge_base"]:
                        continue_save = True
                        st.write("知识库检索输入: ")
                        st.code(st.session_state["rag_tool_calls"][-1]["query"],
                                wrap_lines=True)
                st.write("知识库检索结果：")
                try:
                    # 增加容错，防止 JSON 解析失败
                    content_json = json.loads(event[0].content)
                    # LlamaIndex 返回的通常是直接的字符串，如果是 JSON 格式才解析
                    # 如果你的 LlamaIndex Tool 返回的是纯文本，这里可能会报错，
                    # 建议改为直接显示 content
                    if isinstance(content_json, dict):
                        for k, content in content_json.items():
                            st.write(f"- {k}:")
                            st.code(content, wrap_lines=True)
                    else:
                        st.write(event[0].content)
                except:
                     st.write(event[0].content)

                s.update(label="已完成知识库检索！", expanded=False)
                if continue_save:
                    st.session_state["rag_tool_calls"][-1]["status"] = "已完成知识库检索！"
                    st.session_state["rag_tool_calls"][-1]["content"] = event[0].content
                else:
                    st.session_state["rag_tool_calls"].append(
                        {
                            "status": "已完成知识库检索！",
                            "knowledge_base": event[0].name.replace("_knowledge_base_tool", ""),
                            "content": event[0].content
                        })

def get_rag_graph(platform, model, temperature, selected_kbs, KBS, base_url="", api_key=""):
    try:
        # --- 修改点 2：直接调用 utils.get_chatllm ---
        # 这样就复用了 utils.py 里写好的 DashScope/Ollama/OpenAI 逻辑
        # 只要 utils.py 没问题，这里就一定没问题
        llm = get_chatllm(
            platform_type=platform,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature
        )
        # ------------------------------------------

        # 构建工具列表
        tools = []
        for kb in selected_kbs:
            if kb in KBS:
                tools.append(KBS[kb])

        if not tools:
            raise ValueError("没有有效的知识库工具")
            pass

        # 定义代理函数
        def agent(state):
            messages = state["messages"]
            # 绑定工具 (即使 tools 为空，bind_tools 通常也能处理，但最好判空)
            if tools:
                llm_with_tools = llm.bind_tools(tools)
                response = llm_with_tools.invoke(messages)
            else:
                response = llm.invoke(messages)
            return {"messages": [response]}

        # 创建工作流
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent)
        
        if tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges(
                "agent",
                tools_condition,
                {"tools": "tools", END: END}
            )
            workflow.add_edge("tools", "agent")
        else:
            # 如果没有工具，直接结束
            workflow.add_edge("agent", END)
            
        workflow.set_entry_point("agent")

        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        return app

    except Exception as e:
        st.warning(f"构建 RAG 图失败：{e}")
        return None


def display_chat_history():
    for message in st.session_state["rag_chat_history_with_tool_call"]:
        with st.chat_message(message["role"], avatar=get_img_base64("robot.png") if message["role"] == "assistant" else None):
            if "tool_calls" in message.keys():
                for tool_call in message["tool_calls"]:
                    with st.status(tool_call["status"], expanded=False):
                        st.write("已调用 `", tool_call["knowledge_base"], "` 知识库进行查询")
                        if "query" in tool_call.keys():
                            st.write("知识库检索输入: ")
                            st.code(tool_call["query"], wrap_lines=True)
                        st.write("知识库检索结果：")
                        # 简化显示逻辑，适配 LlamaIndex 的返回
                        content = tool_call.get("content") # 使用 .get() 安全获取
                        if content:
                            st.write(content)
                        else:
                            st.warning("⚠️ 工具调用中断或未返回结果")
            st.write(message["content"])

def clear_chat_history():
    st.session_state["rag_chat_history"] = [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]
    st.session_state["rag_chat_history_with_tool_call"] = [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]
    st.session_state["rag_tool_calls"] = []

def get_rag_chat_response(platform, model, temperature, input, selected_tools, KBS, base_url="", api_key=""):
    app = get_rag_graph(platform, model, temperature, selected_tools, KBS, base_url, api_key)
    if app is None:
        return []
    return graph_response(graph=app, input=input)

def rag_chat_page():
    kbs = get_kb_names()

    KBS = dict()
    kb_root = Path(__file__).resolve().parents[1] / "kb"
    
    # --- ✅ 修改点 3：使用缓存加载工具 ---
    # 遍历加载知识库工具
    for k in kbs:
        kb_path = kb_root / k
        # 这里改用了 load_kb_tool (带缓存)，而不是直接调用 get_llamaindex_tool
        tool = load_kb_tool(k, kb_path)
        if tool:
            KBS[k] = tool
        else:
            # 这里可以改用 toast 或者 info，避免 warning 太多太吵
            # st.warning(f"知识库 {k} 加载失败，已跳过。")
            pass
    # ----------------------------------

    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    if "rag_chat_history_with_tool_call" not in st.session_state:
        st.session_state["rag_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    if "rag_tool_calls" not in st.session_state:
        st.session_state["rag_tool_calls"] = []

    # 侧边栏配置
    with st.sidebar:
        selected_kbs = st.sidebar.multiselect("请选择知识库", kbs)
        
        st.divider()
        st.subheader("平台配置")
        base_url = st.text_input("请输入平台的 Base URL", help="如 Ollama 或 DashScope 的 base_url", key="base_url")
        api_key = st.text_input("请输入 API Key", help="DashScope/OpenAI API Key", type="password", key="api_key")

    display_chat_history()

    cols = st.columns([1.2, 10, 1])
    with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
        platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
        # 获取模型列表
        llm_models = get_llm_models(platform, st.session_state.get("base_url", ""), st.session_state.get("api_key", ""))
        
        if not llm_models:
            llm_models = ["加载失败或列表为空"]
        
        model = st.selectbox("请选择要使用的模型", llm_models)
        temperature = st.slider("请选择模型 Temperature", 0.1, 1., 0.1)
        history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        
    input = cols[1].chat_input("请输入您的问题")
    cols[2].button(":wastebasket:", help="清空对话", on_click=lambda: st.session_state.update({
        "rag_chat_history": [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}],
        "rag_chat_history_with_tool_call": [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}],
        "rag_tool_calls": []
    }))
    
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["rag_chat_history"] += [{"role": 'user', "content": input}]
        st.session_state["rag_chat_history_with_tool_call"] += [{"role": 'user', "content": input}]

        stream_response = get_rag_chat_response(
            platform,
            model,
            temperature,
            st.session_state["rag_chat_history"][-history_len:],
            selected_kbs,
            KBS,
            st.session_state.get("base_url", ""),
            st.session_state.get("api_key", "")
        )

        with st.chat_message("assistant", avatar=get_img_base64("robot.png")):
            response = st.write_stream(stream_response)
        
        st.session_state["rag_chat_history"] += [{"role": 'assistant', "content": response}]
        
        # 将本次对话的工具调用记录合并保存
        st.session_state["rag_chat_history_with_tool_call"] += [{
            "role": 'assistant', 
            "content": response, 
            "tool_calls": st.session_state["rag_tool_calls"]
        }]
        # 清空临时工具调用列表，为下一轮对话做准备
        st.session_state["rag_tool_calls"] = []