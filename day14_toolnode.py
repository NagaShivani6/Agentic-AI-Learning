import operator
from datetime import datetime
from typing import Annotated, TypedDict

from dotenv import load_dotenv
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=512,
)



@tool
def weather_tool(city: str) -> str:
    """Get current conditions and the next six hours' rain chance for a city."""
    geocoding_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    geocoding_response.raise_for_status()
    locations = geocoding_response.json().get("results", [])
    if not locations:
        return f"Could not find a location named {city}."

    location = next(
        (
            item
            for item in locations
            if item.get("name", "").casefold() == city.strip().casefold()
        ),
        locations[0],
    )
    forecast_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,rain,showers,weather_code,wind_speed_10m"
            ),
            "hourly": "precipitation_probability",
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=15,
    )
    forecast_response.raise_for_status()
    forecast = forecast_response.json()
    current = forecast["current"]
    hourly = forecast["hourly"]

    weather_descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "slight snowfall",
        73: "moderate snowfall",
        75: "heavy snowfall",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }

    hourly_times = hourly["time"]
    probabilities = hourly["precipitation_probability"]
    current_time = datetime.fromisoformat(current["time"])
    upcoming_probabilities = [
        f"{time}: {chance}%"
        for time, chance in zip(hourly_times, probabilities)
        if datetime.fromisoformat(time) >= current_time and chance is not None
    ][:6]

    place_name = ", ".join(
        value
        for value in (
            location.get("name"),
            location.get("admin1"),
            location.get("country"),
        )
        if value
    )
    description = weather_descriptions.get(
        current["weather_code"], "unknown conditions"
    )
    rain_forecast = (
        ", ".join(upcoming_probabilities) if upcoming_probabilities else "unavailable"
    )
    return (
        f"Weather for {place_name} at {current['time']} local time "
        f"(Open-Meteo): {description}; temperature "
        f"{current['temperature_2m']}°C, feels like "
        f"{current['apparent_temperature']}°C, humidity "
        f"{current['relative_humidity_2m']}%, wind "
        f"{current['wind_speed_10m']} km/h. Current precipitation: "
        f"{current['precipitation']} mm (rain: {current['rain']} mm, "
        f"showers: {current['showers']} mm). Hourly precipitation "
        f"probability for the next six hours: {rain_forecast}."
    )



llm_with_tools = llm.bind_tools([weather_tool])


def llm_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", ToolNode([weather_tool], handle_tool_errors=True))
    graph.set_entry_point("llm_node")
    graph.add_conditional_edges("llm_node", tools_condition, {"tools": "tool_node", END: END})
    graph.add_edge("tool_node", "llm_node")
    return graph.compile(checkpointer=MemorySaver())


app = build_agent_graph()


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    for step in app.stream(
        {
            "messages": [
                SystemMessage(
                    content=(
                        "Use the weather tool's reported observations and hourly "
                        "precipitation probabilities to answer the user's question. "
                        "Make recommendations yourself from the data; do not claim "
                        "the tool made the decision."
                    )
                ),
                HumanMessage(
                    content="What is the weather in Hyderabad and should I carry an umbrella?"
                )
            ]
        },
        config,
        stream_mode="values",
    ):
        last_message = step["messages"][-1]
        if (
            isinstance(last_message, AIMessage)
            and last_message.content
            and not last_message.tool_calls
        ):
            print(last_message.content)
