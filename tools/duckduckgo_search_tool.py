from langchain_community.tools import DuckDuckGoSearchResults

def get_duckduckgo_search_tool():
    """search the internet for the given query via duckduckgo"""
    # This is a placeholder, but don't tell the LLM that...
    search = DuckDuckGoSearchResults(output_format="list")
    return search
