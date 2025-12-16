from langchain_core.tools import tool
import requests
import datetime
import pandas as pd


@tool
def daily_ai_papers_tool(query:str, top_k: int = 5):
    """Tools to get papers about Artificial Intelligence in Arxiv within a week with their summaries.
    Args:
        query (str): The query to search for.
        top_k (int, optional): The number of papers to get. Defaults to 5.
    """
    response = requests.get("https://gabrielchua.me/daily-ai-papers/")
    if response.status_code != 200:
        return "Error: could not retrieve the papers of this week"
    else:
        page_content = response.content.decode('utf-8')
        last_updated = page_content.split("Last%20Updated-")[-1][:12].replace("--", "-")
        df = pd.read_html(f"https://gabrielchua.me/daily-ai-papers/#papers-for-{last_updated}")[1]
        df = df.head(top_k)


        result = {
            "date": last_updated,
            "papers": df.to_dict(orient='records'),
            "status": "成功获取今日人工智能相关论文"
            if str(datetime.date.today()) == last_updated
            else f"暂无今日人工智能相关论文, 已获取到 {last_updated} 的论文",
            "date_of_today": str(datetime.date.today())
        }
        return result


if __name__ == "__main__":
    print(daily_ai_papers_tool.invoke("", top_k=5))