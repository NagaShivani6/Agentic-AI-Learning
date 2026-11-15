from datetime import datetime
from uuid import uuid4
import ast
import operator

from dotenv import load_dotenv
import requests
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
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
    name="research_agent",
)

@tool
def math_tool(expression: str) -> str:
    """A tool for performing mathematical calculations."""
    try:
        binary_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        unary_operators = {ast.UAdd: operator.pos, ast.USub: operator.neg}

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
                return binary_operators[type(node.op)](
                    evaluate(node.left), evaluate(node.right)
                )
            if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operators:
                return unary_operators[type(node.op)](evaluate(node.operand))
            raise ValueError("Unsupported mathematical expression")

        result = evaluate(ast.parse(expression, mode="eval"))
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
    name="math_agent",
)

@tool
def ask_research_agent(task: str) -> str:
    """Delegate a weather or factual-research request to the research specialist.

    Use this tool when the user asks about weather, current conditions, forecasts,
    locations, or general factual information that should be answered by the
    research agent. Pass only the research question as a single string. The
    research agent will determine whether it needs to call the weather tool and
    return the final answer for that research-only request.
    """
    try:
        result = research_agent.invoke({"messages": [HumanMessage(content=task)]})
        return result["messages"][-1].content
    except Exception as e:
        return f"The research agent encountered an error: {str(e)}"

@tool
def ask_math_agent(task: str) -> str:
    """Delegate a math or arithmetic request to the arithmetic specialist.

    Use this tool for calculations such as addition, subtraction, multiplication,
    division, percentages, powers, modulo, or any other numeric expression. Pass
    a clear arithmetic task or expression as a string. The math agent will
    evaluate the calculation and return only the result for that calculation.
    """
    try:
        result = math_agent.invoke({"messages": [HumanMessage(content=task)]})
        return result["messages"][-1].content
    except Exception as e:
        return f"The math agent encountered an error: {str(e)}"

agent = create_agent(
    llm,
    tools=[ask_research_agent, ask_math_agent],
    system_prompt=(
        "You are a generalist agent. You can handle both weather/factual-research "
        "and arithmetic tasks. Use ask_research_agent for weather requests and ask_math_agent "
        "for calculations. If the request involves both, delegate to the appropriate "
        "specialist agent. Return only the relevant results."
    ),
    checkpointer=MemorySaver(),
    name="general_agent",
)

if __name__ == "__main__":
    thread_id = str(uuid4())
    print(f"Thread: {thread_id}")
    seen = set()

    for update in agent.stream(
        {"messages": [HumanMessage(content="what is 100 divided by 4?")]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 12},
        stream_mode="updates",
    ):
        for node, state_delta in update.items():
            print(f"\n[{node}]")
            for message in state_delta.get("messages", []):
                if message.id in seen:
                    continue
                seen.add(message.id)
                for tool_call in getattr(message, "tool_calls", []):
                    print(f"Routes to {tool_call['name']}: {tool_call['args']}")
                if message.content:
                    print(message.content)