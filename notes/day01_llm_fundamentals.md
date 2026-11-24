# Day 1: LLM Fundamentals

## Tokens
A token is a chunk of text that an LLM processes. A prompt is split into tokens before it is sent to the model. Tokenization is model-specific and uses a vocabulary-based method such as BPE (Byte Pair Encoding), so it is not just splitting on spaces. A word can be one token or several. Code and numbers often tokenize into more tokens than plain English. Cost and context limits are measured in tokens, not words.

## Context window
The context window is the maximum number of tokens the model can handle in one call, input and output combined. Everything counts toward it: the system prompt, the conversation history, tool schemas, tool results, agent reasoning steps and the model's own output. In a long-running agent the history grows quickly. When it overflows, earlier context is dropped and the agent starts to forget what it was doing and makes errors.

## Temperature
Temperature controls how deterministic the model's token selection is. At temperature 0 the model always picks the most probable next token. That is the most deterministic, not the most accurate: the model can still be consistently wrong. Higher temperature adds randomness, which makes output more creative but less consistent. For agents that make tool calls, use a low temperature (0 or close to it).

## Stateless inference
An LLM has no memory between API calls. What feels like memory in a chat is the application re-sending the full conversation history on every call. For example, the third message sends message 1, response 1, message 2, response 2 and message 3. This is why context windows fill up, and why agent memory (short-term in the context, long-term in an external store, or summarized history) is an engineering problem.

## Hallucination vs wrong tool choice
Hallucination means the model generates something false or nonexistent. Ambiguous tool selection means the model picks a real tool but the wrong one, usually because tool descriptions are vague or overlap.
