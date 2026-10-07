from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

@tool
def tool_a(expression: str) -> str:
    """A useful tool."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def tool_b(city: str) -> str:
    """
    just a tool
    """
    weather_data = {
        "hyderabad": "Rainy, 28°C, 80% chance of rain",
        "mumbai": "Humid, 32°C, partly cloudy",
        "bangalore": "Pleasant, 24°C, clear skies",
        "delhi": "Hot, 38°C, sunny"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

tools = [tool_a, tool_b]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=1
)

# Test 1 — calculator
response = agent_executor.invoke({
    "input": "What is 25 multiplied by 4 and then divided by 2?"
})
print(response["output"])

# Test 2 — weather
response = agent_executor.invoke({
    "input": "What is the weather in Hyderabad? Should I carry an umbrella?"
})
print(response["output"])

# Test 3 — both tools
response = agent_executor.invoke({
    "input": "What is the weather in Mumbai and what is 100 divided by 4?"
})
print(response["output"])