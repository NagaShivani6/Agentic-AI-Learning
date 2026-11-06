# 1. Define get_weather and send_email with @tool (clear descriptions!)
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
from langchain.agents import create_agent


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

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email with the given subject and body to the specified recipient."""
    # For demonstration purposes, we'll just return a string instead of actually sending an email.
    return f"Email sent to {to} with subject '{subject}' and body:\n{body}"


# 2. Create the agent with a checkpointer AND an interrupt
#    hint: create_agent / create_react_agent accept an interrupt_before
#    argument. Which node name do you pause before?
#    (Your printed graph showed the names: model and tools.)
agent = create_agent(llm,[get_weather,send_email],checkpointer=MemorySaver(),interrupt_before=["tools"])

# 3. Run the first turn with a thread_id, using stream or invoke
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    agent.invoke(
        {
            "messages": [
                SystemMessage(
                    content=(
                        "Use get_weather first to retrieve current conditions and "
                        "hourly precipitation probabilities. Then summarize the "
                        "weather and call send_email with that summary and the "
                        "requested recipient. Do not claim the email was sent until "
                        "send_email returns."
                    )
                ),
                HumanMessage(
                    content=(
                        "Check the weather in Hyderabad and email the summary "
                        "to test@example.com."
                    )
                ),
            ]
        },
        config,
    )

    while True:
        state = agent.get_state(config)
        if "tools" in state.next:
            last_message = state.values["messages"][-1]
            tool_calls = getattr(last_message, "tool_calls", [])
            print("Agent paused before running these tool calls:")
            for tool_call in tool_calls:
                print(f"- {tool_call['name']}: {tool_call['args']}")

            user_input = input("Approve or reject? (a/r): ").strip().lower()
            if user_input == "a":
                agent.invoke(None, config)
                continue
            if user_input == "r":
                print("Rejected. No pending tools were run.")
                break
            print("Invalid input. Please enter 'a' to approve or 'r' to reject.")
            continue

        final_message = next(
            (
                message
                for message in reversed(state.values.get("messages", []))
                if isinstance(message, AIMessage)
                and message.content
                and not message.tool_calls
            ),
            None,
        )
        if final_message is not None:
            print(final_message.content.encode("ascii", "backslashreplace").decode("ascii"))
        else:
            print("The agent finished without a final text response.")
        break
# 4. Inspect the pause:
#    state = agent.get_state(config)
#    state.next            -> which node is waiting?
#    state.values["messages"][-1].tool_calls  -> what is it about to run?


# 5. input("approve or reject? ")
#    approve -> agent.invoke(None, config)
#    reject  -> decide how to end cleanly (your choice, see below)