from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool

@tool
def arxiv_search_tool(query: str):
    """Searches arxiv.org Articles for the query and returns the articles summaries.
    Args:
        query: The query to search for, should be in English."""
    tool = ArxivAPIWrapper()
    return tool.run(query)

if __name__ == "__main__":
    print(arxiv_search_tool.invoke("Apple Intelligence"))