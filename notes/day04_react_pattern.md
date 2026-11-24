# Day 4: The ReAct Pattern

## What ReAct is
ReAct means Reasoning plus Acting. The LLM alternates between thinking and acting in a loop:
Thought, then Action (a tool call), then Observation (the tool result), then Thought again, until it has enough information to give a final answer.

Example: asked for the cause and date of World War 1, the agent thinks it needs the cause, searches for it, reads the result, thinks it still needs the date, searches again, reads that result, and then writes the final answer.

## Who does what
The reasoning (Thought) and the decision to act happen in the LLM. The Action is the LLM outputting a tool call. The application only executes the tool and returns the result, with no reasoning. The Observation is the LLM reading the tool result from the conversation history.

## Connection to tool calling
Tool calling is the mechanism that makes the Action and Observation steps possible. ReAct is the reasoning pattern built on top of it.

## Max iterations
The loop repeats as many times as the task needs, up to a maximum iterations limit set by the developer. Without a limit an agent can loop forever, fill the context window, run up API costs on every iteration, and repeatedly hammer a failing tool. A limit that is too low stops the agent before it finishes a multi-step task.

## Beyond ReAct
Plan-and-Execute makes a full plan first and then runs each step. Reflexion lets the agent critique its own mistakes and retry. Tree of Thoughts explores several reasoning paths. LangGraph agents put a developer-defined graph around ReAct-style reasoning inside nodes.
