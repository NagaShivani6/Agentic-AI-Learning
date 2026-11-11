# 1. Define get_weather and send_email with @tool (clear descriptions!)
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
import requests
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=512,
)

@tool
def get_weather(city: str) -> str:
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

research_agent = create_agent(
    llm,
    tools=[get_weather],
    system_prompt=(
        "You are the weather and factual-research specialist. Handle only weather "
        "or factual-research parts of the user's request. For weather, call "
        "get_weather and report its result. Do not solve arithmetic or answer "
        "unrelated parts, even if they appear in the shared conversation history. "
        "Return only your findings for the part assigned to you."
    ),
    checkpointer=MemorySaver(),
    name="research_agent",
)

@tool
def math_tool(expression: str) -> str:
    """A tool for performing mathematical calculations."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

math_agent = create_agent(
    llm,
    tools=[math_tool],
    system_prompt=(
        "You are the arithmetic specialist. Handle only calculation parts of the "
        "user's request. Use math_tool for every calculation. Do not answer weather "
        "or factual-research questions, even if they appear in the shared "
        "conversation history. Return only the calculation and its result."
    ),
    checkpointer=MemorySaver(),
    name="math_agent",
)

# The supervisor sees the shared message history and routes work to an agent.

main_agent = create_supervisor(
    agents=[research_agent, math_agent],
    model=llm,
    prompt=(
        "You are a dispatcher, not a specialist. Inspect the latest user request. "
        "For a calculation, delegate that part to math_agent exactly once; never "
        "calculate it yourself. For weather or factual research, delegate that part "
        "to research_agent exactly once; never answer it yourself. If both are "
        "requested, call each relevant specialist once, in either order. Do not send "
        "the same task to an agent again after it has replied. After all required "
        "specialists have replied, combine only their findings into one concise "
        "answer and stop. Do not treat an agent's handoff-back message as a new "
        "user request."
    ),
    output_mode="last_message",
).compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    thread_id = str(uuid4())
    print(f"Thread: {thread_id}")
    for update in main_agent.stream(
        {"messages": [HumanMessage(content="What’s the weather in Mumbai, and what is 100 divided by 4?")]},
        {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 12,
        },
        stream_mode="updates",
    ):
        for node, state_delta in update.items():
            print(f"\n[{node}]")
            for message in state_delta.get("messages", []):
                tool_calls = getattr(message, "tool_calls", [])
                if tool_calls:
                    for tool_call in tool_calls:
                        print(
                            f"Supervisor routes to "
                            f"{tool_call['name']}: {tool_call['args']}"
                        )
                if message.content:
                    print(message.content)

        