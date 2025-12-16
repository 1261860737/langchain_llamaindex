from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
@tool
def serpapi_search_tool(query: str):
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具。"""
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    if not serp_api_key:
        return "错误：未设置 SERPAPI_API_KEY 环境变量"
    
    serp = SerpAPIWrapper(
    serpapi_api_key=serp_api_key,
    params={
        "hl": "zh-cn",
        "gl": "cn",
        "num": 3
    },
    search_engine="baidu"
)
    result = serp.run(query)
    return result
