"""
FastAPI app for the SHL Assessment Recommender API.
Stateless: full conversation history passed in every request.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from agent import run_agent

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational API for recommending SHL assessments based on hiring needs.",
    version="1.0.0",
)

# CORS — allow all origins for broad API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — always returns 200 with status ok."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts full conversation history,
    returns agent response with optional recommendations.
    """
    # Validate: max 8 turns
    if len(request.messages) >= 8:
        return ChatResponse(
            reply="We've reached the conversation limit. Here are your final recommendations.",
            recommendations=[],
            end_of_conversation=True,
        )

    # Call agent with 25 second timeout
    try:
        # Convert Pydantic models to dicts for the agent
        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

        result = await asyncio.wait_for(
            run_agent(messages_dicts),
            timeout=25.0,
        )

        return ChatResponse(
            reply=result["reply"],
            recommendations=[
                Recommendation(**rec) for rec in result["recommendations"]
            ],
            end_of_conversation=result["end_of_conversation"],
        )

    except asyncio.TimeoutError:
        return ChatResponse(
            reply="Request timed out. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        return ChatResponse(
            reply="Something went wrong. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )


# ── Local dev entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
