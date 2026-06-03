import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()

# llama3.2 (3B) generates ~36 tokens/s vs llama3.1:8b at ~18 tokens/s.
# num_predict caps the output so the model doesn't ramble — enough for a
# clear demo answer without a 30-second wait.
llm = ChatOllama(model="llama3.2:latest", temperature=0.7, num_predict=250)

SYSTEM_PROMPT = """You are a helpful career advisor for students in Hawaii interested in IT and cybersecurity careers.
Answer questions about career paths, certifications, and educational options.
Be concise — 2 to 4 sentences unless the question clearly requires more."""


class QueryRequest(BaseModel):
    question: str


async def _token_stream(messages: list):
    """Yield SSE-formatted token events, then a [DONE] sentinel."""
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/query")
async def vanilla_query(request: QueryRequest):
    """
    Mode A: Vanilla LLM — no retrieval, no graph. Streams tokens as they arrive.
    The model answers from training data alone and will hallucinate PCATT-specific details.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request.question),
    ]
    return StreamingResponse(
        _token_stream(messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
