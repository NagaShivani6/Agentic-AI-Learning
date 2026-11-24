import faiss
from pathlib import Path
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=512,
)
NOTES_DIR = Path(__file__).parent / "notes"

def chunk_text(text, size=800, overlap=100):
    """Chunk the text into smaller pieces of a given size with overlap."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be positive and overlap must be between 0 and size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


all_chunks = []
for path in sorted(NOTES_DIR.glob("*.md")):
    text = path.read_text(encoding="utf-8")
    all_chunks.extend((path.name, chunk) for chunk in chunk_text(text))

chunk_texts = [chunk for _, chunk in all_chunks]
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunk_texts, normalize_embeddings=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

RELEVANCE_THRESHOLD = 0.25


def top_score(query: str) -> float:
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, _ = index.search(query_embedding, 1)
    return float(scores[0][0])


def retrieve(query: str, k=3):
    """retrieve the top k most relevant chunks for a query, along with their source file names and similarity scores and returns an empty list when the best score is below the threshold"""
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, min(k, len(all_chunks)))
    kept = [
        (*all_chunks[i], float(s))
        for s, i in zip(scores[0], indices[0])
        if s >= RELEVANCE_THRESHOLD
    ]
    return kept

@tool
def search_notes(query: str) -> str:
    """Search my study notes on LLM basics, tool calling, ReAct, LangGraph state,
    persistence and human-in-the-loop. Pass a short question or keywords. Returns
    the top excerpts with their source file names."""
    results = retrieve(query)
    if not results:
        return "No relevant excerpts found"
    return "\n\n".join(f"[{src}] {chunk}" for src, chunk, _ in results)
    

agent = create_agent(
    model = llm,
    tools=[search_notes],
    system_prompt=(
        "Answer only from the excerpts returned by search_notes. Name the source files you used.If the excerpts do not answer the question, say the notes do not cover it. Do not use outside knowledge."
    ),
    checkpointer=MemorySaver(),
)

def ask(question: str) -> None:
    thread_id = str(uuid4())
    seen = set()
    print(f"\n=== {question} ===")
    for update in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 12},
        stream_mode="updates",
    ):
        for node, delta in update.items():
            for message in delta.get("messages", []):
                if message.id in seen:
                    continue
                seen.add(message.id)
                for tc in getattr(message, "tool_calls", []):
                    print(f"Routes to {tc['name']}: {tc['args']}")
                if message.content:
                    text = message.content
                    if node == "tools":
                        text = text[:200] + "..."
                    print(f"[{node}] {text}")


if __name__ == "__main__":
    print(f"Number of chunks: {len(all_chunks)}")
    print("First chunk:", all_chunks[0])
    print(f"Embeddings shape: {embeddings.shape}")

    in_corpus_questions = [
        "What does operator.add do?",
        "How does the ReAct loop use tool calling?",
        "What does a checkpointer save, and how is thread_id used?",
    ]
    out_of_corpus_questions = [
        "What is the capital of France?",
        "What is the weather in Mumbai today?",
        "Who won the 2022 FIFA World Cup?",
    ]

    in_corpus_scores = []
    out_of_corpus_scores = []
    for label, questions, scores in (
        ("In-corpus", in_corpus_questions, in_corpus_scores),
        ("Out-of-corpus", out_of_corpus_questions, out_of_corpus_scores),
    ):
        print(f"\n{label} top-1 scores:")
        for question in questions:
            score = top_score(question)
            scores.append(score)
            print(f"{score:.4f} | {question}")

    min_in_corpus = min(in_corpus_scores)
    max_out_of_corpus = max(out_of_corpus_scores)
    print(
        "\nScore ranges: "
        f"in-corpus {min(in_corpus_scores):.4f}-{max(in_corpus_scores):.4f}, "
        f"out-of-corpus {min(out_of_corpus_scores):.4f}-{max_out_of_corpus:.4f}"
    )
    print(
        "The groups separate cleanly."
        if max_out_of_corpus < min_in_corpus
        else "The groups overlap; a single relevance threshold is unreliable."
    )

    print('\nTop 3 results for "What does operator.add do?":')
    for source_file, chunk, score in retrieve("What does operator.add do?"):
        print(f"\nSource: {source_file} | Score: {score:.4f}\n{chunk}")

    for q in [
        "What does operator.add do in AgentState?",
        "What is the capital of France?",
        "How does tool calling relate to the ReAct loop?",
    ]:
        ask(q)