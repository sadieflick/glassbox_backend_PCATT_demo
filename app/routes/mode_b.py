import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()

CHROMA_DIR = Path(__file__).parent.parent.parent / "chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

_pcatt = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="pcatt_courses",
)
_nice = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="nice_work_roles",
)

llm = ChatOllama(model="llama3.2:latest", temperature=0.3, num_predict=300)

SYSTEM_PROMPT = """You are a helpful career advisor for students in Hawaii interested in IT and cybersecurity careers.
You have been given two types of reference material:
  1. PCATT COURSES — specific courses offered by the Pacific Center for Advanced Technology Training
  2. NICE WORK ROLES — career role definitions from the NICE Cybersecurity Workforce Framework, including aligned industry certifications and required knowledge/skills/tasks

Answer the user's question using ONLY the provided reference material. Be specific and concise.
If the material doesn't fully answer the question, say so — do not invent courses, roles, or certifications."""


class QueryRequest(BaseModel):
    question: str


def _nice_available() -> bool:
    try:
        return _nice._collection.count() > 0
    except Exception:
        return False


def _excerpt(page_content: str, max_chars: int = 280) -> str:
    """
    Return a readable excerpt from a ChromaDB page_content string.
    Skips the first paragraph (which repeats the title/ID already shown in the card)
    and takes up to max_chars of the remaining body.
    """
    parts = page_content.split("\n\n", 1)
    body = parts[1].strip() if len(parts) > 1 else page_content
    return body[:max_chars] + ("…" if len(body) > max_chars else "")


def _scores(raw: float) -> tuple[float, float]:
    """
    Return (distance, similarity_pct) from a raw ChromaDB score.

    ChromaDB returns cosine distance in [0, 2] (0 = identical).
    We pass the raw value as 'distance' and derive a [0, 100] percentage
    via: similarity = (1 - distance/2) * 100, clamped to [0, 100].
    """
    distance = round(raw, 4)
    similarity_pct = round(max(0.0, min(100.0, (1.0 - raw / 2.0) * 100)), 1)
    return distance, similarity_pct


async def _rag_stream(question: str) -> AsyncGenerator[str, None]:
    """
    Retrieve → emit sources → stream tokens.

    Sending sources as the first SSE event lets the frontend display
    'where this answer is coming from' before the first token arrives.
    Each source is a structured dict (not a flat string) so the frontend
    can render distance scores, excerpts, and metadata without re-parsing.
    """
    # 1. Retrieval with scores (fast — pure vector math, no LLM involved)
    pcatt_results = _pcatt.similarity_search_with_score(question, k=3)

    context_blocks: list[str] = []
    sources: list[dict] = []

    for doc, raw_score in pcatt_results:
        cid   = doc.metadata.get("course_id", "Unknown")
        title = doc.metadata.get("title", "Unknown")
        distance, similarity_pct = _scores(raw_score)
        sources.append({
            "prefix":         "PCATT",
            "id":             cid,
            "title":          title,
            "source_type":    "course",
            "distance":       distance,
            "similarity_pct": similarity_pct,
            "excerpt":        _excerpt(doc.page_content),
            "dept":           doc.metadata.get("dept", ""),
        })
        context_blocks.append(f"[PCATT COURSE: {cid}]\n{doc.page_content}")

    if _nice_available():
        nice_results = _nice.similarity_search_with_score(question, k=2)
        for doc, raw_score in nice_results:
            rid   = doc.metadata.get("work_role_id", "Unknown")
            title = doc.metadata.get("title", "Unknown")
            distance, similarity_pct = _scores(raw_score)
            sources.append({
                "prefix":         "NICE",
                "id":             rid,
                "title":          title,
                "source_type":    "work_role",
                "distance":       distance,
                "similarity_pct": similarity_pct,
                "excerpt":        _excerpt(doc.page_content),
                "category":       doc.metadata.get("category", ""),
                "certifications": doc.metadata.get("certifications", ""),
            })
            context_blocks.append(f"[NICE WORK ROLE: {rid}]\n{doc.page_content}")

    # 2. Emit sources immediately — frontend can show these before the LLM starts
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # 3. Stream LLM tokens
    context = "\n\n---\n\n".join(context_blocks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Reference material:\n\n{context}\n\nQuestion: {question}"),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/query")
async def rag_query(request: QueryRequest):
    """
    Mode B: Vector RAG — retrieve from ChromaDB, stream sources then tokens.
    Sources arrive before the first word so the audience sees the evidence first.
    """
    return StreamingResponse(
        _rag_stream(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
