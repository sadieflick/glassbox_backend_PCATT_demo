from fastapi import APIRouter
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()

# This is the LangChain object that talks to your local Ollama instance.
# "ChatOllama" knows to send requests to http://localhost:11434 by default.
llm = ChatOllama(model="llama3.1:8b", temperature=0.7)

SYSTEM_PROMPT = """You are a helpful career advisor for students in Hawaii interested in IT and cybersecurity careers.
Answer questions about career paths, certifications, and educational options."""


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    mode: str
    sources: list[str]


@router.post("/query", response_model=QueryResponse)
async def vanilla_query(request: QueryRequest):
    """
    Mode A: Vanilla LLM — no retrieval, no graph.
    The model answers from its training data alone.
    It has no knowledge of specific PCATT courses, so it will hallucinate or give generic answers.
    This is the 'fail' baseline for the demo.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request.question),
    ]

    response = await llm.ainvoke(messages)

    return QueryResponse(
        answer=response.content,
        mode="vanilla_llm",
        sources=[],  # No sources — that's the point
    )
