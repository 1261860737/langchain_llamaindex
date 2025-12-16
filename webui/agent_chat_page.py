import streamlit as st
from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
# from tools.llamaindex_tool import get_llamaindex_tool
from tools import (
    weather_search_tool,
    get_duckduckgo_search_tool,
    arxiv_search_tool,
    wikipedia_search_tool,
    daily_ai_papers_tool,
    serpapi_search_tool,
    get_llamaindex_tool,
)
import os
from pathlib import Path

AGENT_PAGE_INTRODUCTION = "你好，我是你的智能助手，当前页面为`Agent 对话模式`，可以在对话让大模型借助左侧所选工具进行回答，有什么可以帮助你的吗？"

def get_agent_graph(platform, model, temperature, selected_tools, TOOLS):
    tools = [TOOLS[k] for k in selected_tools]
    tool_node = ToolNode(tools=tools)

    def call_model(state):
        llm = get_chatllm(platform, model, temperature=temperature)
        llm_with_tools = llm.bind_tools(tools)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


def graph_response(graph, input):
    for event in graph.stream(
        {"messages": input},
        config={"configurable": {"thread_id": 42}},
        stream_mode="messages",
    ):
        # st.write(event)
        # st.write(graph.get_state_history(config={"configurable": {"thread_id": 42}},))

        if type(event[0]) == AIMessageChunk:
            if len(event[0].tool_calls):
                # st.write(event[0].tool_calls)
                st.session_state["agent_tool_calls"].append(
                    {
                        "status": "正在调用工具...",
                        "tool": event[0].tool_calls[0]["name"],
                        "args": str(event[0].tool_calls[0]["args"]),
                    }
                )
            yield event[0].content
        elif type(event[0]) == ToolMessage:
            status_placeholder = st.empty()
            with status_placeholder.status("正在调用工具...", expanded=True) as s:
                st.write("已调用 `", event[0].name, "` 工具")  # Show which tool is being called
                continue_save = False
                if len(st.session_state["agent_tool_calls"]):
                    if  "content" not in st.session_state["agent_tool_calls"][-1].keys() \
                            and event[0].name == st.session_state["agent_tool_calls"][-1]["tool"]:
                        continue_save = True
                        st.write("工具输入: ")
                        st.code(st.session_state["agent_tool_calls"][-1]["args"], wrap_lines=True)  # Display the input data sent to the tool
                st.write("工具输出：")
                st.code(event[0].content, wrap_lines=True) # Placeholder for tool output that will be updated later below
                s.update(label="已完成工具调用！", expanded=False)
            if continue_save:
                st.session_state["agent_tool_calls"][-1]["status"] = "已完成工具调用！"
                st.session_state["agent_tool_calls"][-1]["content"] = event[0].content
            else:
                st.session_state["agent_tool_calls"].append(
                    {
                        "status": "已完成工具调用！",
                        "tool": event[0].name,
                        "content": event[0].content
                    })


def get_agent_chat_response(platform, model, temperature, input, selected_tools, TOOLS):
    app = get_agent_graph(platform, model, temperature, selected_tools, TOOLS)
    return graph_response(graph=app, input=input)


def display_chat_history():
    for message in st.session_state["agent_chat_history_with_tool_call"]:
        with st.chat_message(message["role"], avatar=get_img_base64("robot.png") if message["role"] == "assistant" else None):
            if "tool_calls" in message.keys():
                for tool_call in message["tool_calls"]:
                    # 获取状态，如果不存在则默认为完成
                    status = tool_call.get("status", "已完成工具调用")
                    
                    with st.status(status, expanded=False):
                        # 安全获取工具名称
                        tool_name = tool_call.get("tool", "未知工具")
                        st.write("已调用 `", tool_name, "` 工具")
                        
                        if "args" in tool_call.keys():
                            st.write("工具输入: ")
                            st.code(tool_call["args"], wrap_lines=True)
                        
                        st.write("工具输出：")
                        # --- ✅ 修复点：使用 .get() 安全获取 content ---
                        # 如果没有 content (比如上次报错中断了)，显示提示信息而不报错
                        content = tool_call.get("content", "⚠️ 工具调用中断或未返回结果")
                        st.code(content, wrap_lines=True)
                        # -------------------------------------------

            # 同样安全获取 content
            st.write(message.get("content", ""))

def clear_chat_history():
    st.session_state["agent_chat_history"] = [
            {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
        ]
    st.session_state["agent_chat_history_with_tool_call"] = [
        {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
    ]
    st.session_state["agent_tool_calls"] = []


def agent_chat_page():
    kbs = get_kb_names()
    try:
        duckduckgo_search_tool = get_duckduckgo_search_tool()
    except Exception:
        duckduckgo_search_tool = None
    TOOLS = {
        "天气查询": weather_search_tool,
        "Arxiv 搜索": arxiv_search_tool,
        "Wikipedia 搜索": wikipedia_search_tool,
        "今日AI论文查询": daily_ai_papers_tool,
        "Serpapi 搜索": serpapi_search_tool,
    }
    if duckduckgo_search_tool:
        TOOLS["Duckduckgo 搜索"] = duckduckgo_search_tool

    kb_root = Path(__file__).resolve().parents[1] / "kb"
    for k in kbs:
        kb_path = kb_root / k
        
        # 使用 LlamaIndex 加载工具 (替代了原来的 get_advanced_rag_tool)
        tool = get_llamaindex_tool(k, kb_path)
        
        if tool:
            TOOLS[f"{k} 知识库"] = tool
        else:
            st.warning(f"知识库 {k} 加载失败或不存在有效索引，已跳过该知识库。")
    
    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = [
            {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
        ]
    if "agent_chat_history_with_tool_call" not in st.session_state:
        st.session_state["agent_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}
        ]
    if "agent_tool_calls" not in st.session_state:
        st.session_state["agent_tool_calls"] = []

    with st.sidebar:
        selected_tools = st.multiselect("请选择对话中可使用的工具", list(TOOLS.keys()), default=list(TOOLS.keys()))
        

    display_chat_history()

    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            model = st.selectbox("请选择要使用的模型", get_llm_models(platform))
            temperature = st.slider("请选择模型 Temperature", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["agent_chat_history"] += [{"role": 'user', "content": input}]
        st.session_state["agent_chat_history_with_tool_call"] += [{"role": 'user', "content": input}]

        # print(st.session_state["agent_chat_history"][-history_len:])
        stream_response = get_agent_chat_response(
            platform,
            model,
            temperature,
            st.session_state["agent_chat_history"][-history_len:],
            selected_tools,
            TOOLS
        )

        with st.chat_message("assistant", avatar=get_img_base64("robot.png")):
            response1 = st.write_stream(stream_response)
        st.session_state["agent_chat_history"] += [{"role": 'assistant', "content": response1}]
        st.session_state["agent_chat_history_with_tool_call"] += [{"role": 'assistant', "content": response1, "tool_calls": st.session_state["agent_tool_calls"]}]
        st.session_state["agent_tool_calls"] = []
