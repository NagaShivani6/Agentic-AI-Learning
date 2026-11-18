from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()


@dataclass
class MemoryContext:
    user_id: str


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=512,
)

store = InMemoryStore()
checkpointer = MemorySaver()

_STOP_WORDS = {
    "a", "an", "and", "are", "do", "does", "for", "how", "i", "is", "it",
    "me", "my", "of", "please", "that", "the", "to", "was", "what", "when",
    "where", "which", "who", "why", "you", "your",
}


def memory_namespace(user_id: str) -> tuple[str, str]:
    return (user_id, "memories")


def memory_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.casefold())
        if term not in _STOP_WORDS
    }


@tool
def save_memory(fact: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """Save a fact for the current user, without duplicating an existing fact."""
    assert runtime.store is not None
    normalized_fact = " ".join(fact.casefold().split())
    memory_id = hashlib.sha256(normalized_fact.encode("utf-8")).hexdigest()
    namespace = memory_namespace(runtime.context.user_id)

    if runtime.store.get(namespace, memory_id) is not None:
        return "That memory is already saved."

    runtime.store.put(
        namespace,
        memory_id,
        {
            "fact": fact,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return f"Memory saved with ID: {memory_id}"


@tool
def recall_memory(query: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """Find up to five saved memories with keywords matching the query."""
    assert runtime.store is not None
    query_terms = memory_terms(query)
    if not query_terms:
        return "No searchable terms in the query."

    memories = runtime.store.search(memory_namespace(runtime.context.user_id), limit=100)
    ranked_memories = sorted(
        (
            (len(query_terms & memory_terms(item.value["fact"])), item)
            for item in memories
        ),
        key=lambda result: result[0],
        reverse=True,
    )
    matches = [item for score, item in ranked_memories if score > 0][:5]
    if not matches:
        return "No matching memories found."
    return "\n".join(
        f"memory_id={item.key}; fact={item.value['fact']}" for item in matches
    )


@tool
def delete_memory(memory_id: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """Delete a saved memory by its ID after identifying it with recall_memory."""
    assert runtime.store is not None
    namespace = memory_namespace(runtime.context.user_id)
    if runtime.store.get(namespace, memory_id) is None:
        return "No saved memory with that ID was found."
    runtime.store.delete(namespace, memory_id)
    return "Memory deleted."


def main() -> None:
    agent = create_agent(
        model=llm,
        tools=[save_memory, recall_memory, delete_memory],
        system_prompt=(
            "You are a memory agent. Use save_memory when the user asks you "
            "to remember a fact. Use recall_memory to answer questions about "
            "saved facts. To forget a specific fact, use recall_memory to "
            "identify it and then delete_memory with its ID. If the requested "
            "memory is ambiguous, ask which fact to forget before deleting."
        ),
        context_schema=MemoryContext,
        checkpointer=checkpointer,
        store=store,
    )

    def run_turn(prompt: str, thread_id: str, user_id: str) -> None:
        print(f"\nThread: {thread_id}")
        for update in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 12,
            },
            context=MemoryContext(user_id=user_id),
            stream_mode="updates",
        ):
            for node, state_delta in update.items():
                print(f"\n[{node}]")
                for message in state_delta.get("messages", []):
                    for tool_call in getattr(message, "tool_calls", []):
                        print(f"Routes to {tool_call['name']}: {tool_call['args']}")
                    if message.content:
                        print(message.content)

    demo_user_id = "demo-user"
    thread_a = str(uuid4())
    thread_b = str(uuid4())
    thread_a_prompt = "My favorite city is Hyderabad. Remember that."

    run_turn(thread_a_prompt, thread_a, demo_user_id)
    run_turn("What's my favorite city?", thread_b, demo_user_id)

    thread_b_state = agent.get_state(
        {"configurable": {"thread_id": thread_b}}
    )
    thread_b_messages = thread_b_state.values["messages"]
    print("\nThread B checkpoint messages:")
    for message in thread_b_messages:
        print(f"{type(message).__name__}: {message.content}")

    thread_a_prompt_in_thread_b = any(
        thread_a_prompt in str(message.content) for message in thread_b_messages
    )
    print(
        "\nThread A prompt is in Thread B checkpoint: "
        f"{thread_a_prompt_in_thread_b}"
    )


if __name__ == "__main__":
    main()
