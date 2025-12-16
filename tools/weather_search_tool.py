from langchain_core.tools import tool
import requests
import datetime

@tool
def weather_search_tool(city: str):
    """
    Search for global weather using wttr.in (No API Key required).
    Args:
        city: The city to search for. Input should be in English, e.g., "Beijing", "Hangzhou", "London".
    """
    try:
        # wttr.in 不需要查经纬度，直接拼接城市名即可
        # format=j1 表示返回 JSON 格式数据
        url = f"https://wttr.in/{city}?format=j1"
        
        # 设置短超时，如果 5 秒连不上就报错
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            return f"Error: Failed to retrieve weather for {city} from wttr.in (Status: {response.status_code})"
        
        data = response.json()
        
        # 解析数据
        # 1. 获取当前天气
        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "N/A")
        desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
        humidity = current.get("humidity", "N/A")
        
        # 2. 获取今天预报 (wttr.in 返回的 weather 列表就是预报，index 0 是今天)
        today = data.get("weather", [{}])[0]
        date = today.get("date", "N/A")
        max_c = today.get("maxtempC", "N/A")
        min_c = today.get("mintempC", "N/A")
        
        # 3. 格式化输出
        result_str = (
            f"Location: {city}\n"
            f"Date: {date}\n"
            f"Condition: {desc}\n"
            f"Current Temp: {temp_c}°C\n"
            f"Today's High: {max_c}°C\n"
            f"Today's Low: {min_c}°C\n"
            f"Humidity: {humidity}%\n"
            f"Source: wttr.in"
        )
        return result_str

    except Exception as e:
        return f"Error connecting to weather service: {str(e)}"

if __name__ == "__main__":
    # 测试一下
    print(weather_search_tool.invoke("Hangzhou"))