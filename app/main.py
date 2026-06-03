from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import mode_a, mode_b, mode_c

app = FastAPI(title="Glass Box AI — PCATT Demo Backend")

# Allow the React frontend (running on localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Each mode gets its own router with a URL prefix.
# Mode A: POST /mode-a/query
# Mode B: POST /mode-b/query
app.include_router(mode_a.router, prefix="/mode-a", tags=["Vanilla LLM"])
app.include_router(mode_b.router, prefix="/mode-b", tags=["Vector RAG"])
app.include_router(mode_c.router, prefix="/mode-c", tags=["GraphRAG"])


@app.get("/health")
async def health():
    return {"status": "ok"}
