# Day 8: State, Persistence and Human-in-the-Loop

## AgentState and operator.add
In LangGraph the state is a shared TypedDict that every node reads from and writes to. A field annotated as Annotated[list, operator.add] has a reducer: operator.add tells LangGraph how to merge a node's update into the existing value. For the messages field it appends new messages instead of replacing the list, so the full conversation history is kept. Without it, each node would overwrite the previous messages and the agent would lose context. A reducer is different from a checkpointer: the reducer defines how state updates, the checkpointer saves state.

## Checkpointer and thread_id
A checkpointer saves the graph state after every node. Because LLMs are stateless, this gives persistence and lets a run be resumed. MemorySaver keeps state in memory and loses it when the program stops. SqliteSaver and PostgresSaver persist it. The thread_id identifies a conversation: the checkpointer saves and loads state per thread_id, so separate conversations do not mix. A checkpointer is required for human-in-the-loop.

## Streaming
invoke() returns the result once the run is finished. stream() returns output as it is produced, including which node ran and which tool was called. Streaming makes chat interfaces feel responsive and helps debugging.

## Human-in-the-loop (HITL)
With interrupt_before, the graph stops before a named node runs, saves its state to the checkpointer, and waits for a human. To resume, call invoke(None, config) with the same thread_id. Passing None means no new input; passing the original input would add a duplicate message. The human can change the state during the pause with update_state, and the as_node argument makes the update count as if that node produced it. Place an interrupt before expensive or irreversible actions.

## Conditional edges and self-reflection
A conditional edge calls a function that looks at the state and returns the next node (or END). This enables loops, branching and termination. In the essay writer, generate, reflect and research-critique form a loop that revises the essay until it reaches the maximum number of revisions. This is the self-reflection pattern.
