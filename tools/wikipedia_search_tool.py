from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


@tool
def wikipedia_search_tool(query: str):
    """Searches arxiv.org Articles for the query and returns the articles summaries.
    Args:
        query: The query to search for, should be in English."""
    try:
        # 实例化 Wikipedia API 包装器
        # doc_content_chars_max 控制返回内容的长度
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)
        
        # 执行查询
        result = wikipedia.run(query)
        return result

    except Exception as e:
        # 捕获所有网络错误，返回友好的提示信息而不是让程序崩溃
        return f"Wikipedia Search Failed: Connection error or timeout. (Error info: {str(e)})"

if __name__ == "__main__":
    print(wikipedia_search_tool.invoke("Alan Turing"))