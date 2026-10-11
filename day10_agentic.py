import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langchain_community.utilities import RequestsWrapper
from duckduckgo_search import DDGS
from langchain_core.tools import tool


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
)



@tool
def weather_tool(city: str) -> str:
    """Search DuckDuckGo for current weather information for a city and fetch page content."""
    results = DDGS().text(
        f"current weather in {city}",
        max_results=3,
    )
    if not results:
        return f"DuckDuckGo returned no weather results for {city}."

    requests = RequestsWrapper()
    outputs = []

    for result in results:
        title = result.get("title", "")
        body = result.get("body", "")
        href = result.get("href", "")

        # Try fetching the page content
        try:
            page_content = requests.get(href)
            snippet = page_content[:500]  # first 500 chars of HTML
        except Exception as e:
            snippet = f"Could not fetch page: {e}"

        outputs.append(
            f"{title}\n{body}\n{href}\n\nPage snippet:\n{snippet}"
        )

    return "\n\n".join(outputs)




llm_with_tools = llm.bind_tools([weather_tool])


def llm_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        return {"messages": []}

    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]

        if tool_name == "weather_tool":
            result = weather_tool.invoke(tool_input)
            tool_results.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    return {"messages": tool_results}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "tool_node" if getattr(last_message, "tool_calls", []) else END


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.set_entry_point("llm_node")
    graph.add_conditional_edges(
        "llm_node",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END,
        },
    )
    graph.add_edge("tool_node", "llm_node")
    return graph.compile(checkpointer=MemorySaver())


app = build_agent_graph()
config = {"configurable": {"thread_id": "1"}}

for step in app.stream(
    {"messages": [HumanMessage(content="What is the weather in Hyderabad and should I carry an umbrella?")]},
    config,
    stream_mode="values",
):
    last_message = step["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls:
        print(last_message.content.encode("ascii", "backslashreplace").decode("ascii"))
