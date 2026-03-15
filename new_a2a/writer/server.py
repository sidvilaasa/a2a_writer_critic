import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import new_a2a.writer.agent_executor as writer_executor
from new_a2a.a2a_models import AgentCard, AgentCapabilities, AgentSkill, A2ATask, A2ATaskResult


fastapi_app = FastAPI(
    title="Writer Agent (A2A)",
    description=(
        "A2A-compliant server wrapping a LangGraph WriterAgent that generates "
        "and iteratively refines text based on a topic and critic feedback."
    ),
    version="1.0.0",
)

AGENT_CARD = AgentCard(
    name="Writer Agent",
    description=(
        "A LangGraph agent that writes high-quality, engaging text on any topic. "
        "Accepts a topic (and optional critic feedback) and returns a polished piece."
    ),
    url="http://localhost:8001",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    skills=[
        AgentSkill(
            id="creative_writing",
            name="Creative / Informational Writing",
            description=(
                "Write a complete piece on the given topic. "
                "Incorporates any critic feedback to iteratively improve the output."
            ),
            examples=[
                "Write a short story about a robot learning to paint.",
                "Write an article on the benefits of renewable energy.",
                "Write a poem about the ocean at night.",
            ],
        ),
    ],
)


@fastapi_app.get(
    "/.well-known/agent.json",
    response_model=AgentCard,
    summary="Agent Discovery — returns this agent's card",
    tags=["A2A Protocol"],
)
def get_agent_card() -> AgentCard:
    return AGENT_CARD


@fastapi_app.post(
    "/tasks/send",
    response_model=A2ATaskResult,
    summary="Send a writing task to the Writer Agent",
    tags=["A2A Protocol"],
)
def send_task(task: A2ATask) -> A2ATaskResult:
    result = writer_executor.run(task)
    if result.status == "failed":
        raise HTTPException(status_code=500, detail=result.error)
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("  Writer Agent — A2A Server")
    print("  Listening on http://0.0.0.0:8001")
    print("  Agent card: http://localhost:8001/.well-known/agent.json")
    print("  Swagger UI: http://localhost:8001/docs")
    print("=" * 60)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8001)
