# Day 3: Tool Calling

## The round trip
1. The user sends a prompt.
2. The application sends the prompt plus the tool schemas to the LLM.
3. The LLM reads the tool descriptions and decides whether to call a tool.
4. The LLM outputs a structured tool call (tool name, arguments, tool call ID).
5. The application executes the actual function. The LLM never executes anything.
6. The application appends the result to the conversation history as a tool result message and calls the LLM again with the full history.
7. The LLM either calls another tool or writes the final answer.

## Who does what
The LLM decides which tool to call and outputs a payload. The application executes the tool, manages the history and matches results to calls. The LLM sees a JSON schema of each tool (name, description, parameters), never the Python function itself.

## Tool call ID
Each tool call has a unique ID. The result returns with the same ID, so the LLM knows which result belongs to which call. This matters because the LLM can request several tool calls in one response.

## Parallel tool calling
The LLM can request several tools at once in a single response. The application runs them all and returns every result with its matching ID. This reduces the number of LLM calls.

## Multi-step chaining
After seeing a tool result the LLM can call another tool based on that result. The chain continues until it has enough information to give a final answer. This is what makes agents more powerful than a fixed chain.

## Tool descriptions
The description is the only signal the LLM has for choosing a tool. Vague descriptions cause wrong tool selection, especially when tool names are not obvious.
