from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import mode_a

app = FastAPI(title="Glass Box AI — PCATT Demo Backend")

# Allow the React frontend (running on localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Each mode gets its own router with a URL prefix.
# Mode A: POST /mode-a/query
# Mode B: POST /mode-b/query  (coming next)
app.include_router(mode_a.router, prefix="/mode-a", tags=["Vanilla LLM"])


@app.get("/health")
async def health():
    return {"status": "ok"}
