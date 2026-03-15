import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import new_a2a.critic.agent_executor as agent_executor
from new_a2a.a2a_models import AgentCard, AgentCapabilities, AgentSkill, A2ATask, A2ATaskResult

fastapi_app = FastAPI(
    title="Critic Agent (A2A)",
    description=(
        "A2A-compliant server wrapping a LangGraph CriticAgent that evaluates text "
        "and returns actionable feedback, or NO_FURTHER_FEEDBACK when satisfied."
    ),
    version="1.0.0",
)

AGENT_CARD = AgentCard(
    name="Critic Agent",
    description=(
        "A LangGraph agent that evaluates text quality and provides specific, "
        "actionable feedback. Signals NO_FURTHER_FEEDBACK when the text is excellent."
    ),
    url="http://localhost:8002",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=False,
        pushNotifications=False,
        stateTransitionHistory=False,
    ),
    skills=[
        AgentSkill(
            id="text_critique",
            name="Text Critique & Evaluation",
            description=(
                "Evaluates any piece of writing for clarity, engagement, grammar, "
                "and completeness. Returns numbered actionable feedback or "
                "NO_FURTHER_FEEDBACK if the text needs no improvement."
            ),
            examples=[
                "Evaluate this short story for quality and suggest improvements.",
                "Critique this article on renewable energy.",
                "Review this poem and tell me what to improve.",
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
    summary="Send text to the Critic Agent for evaluation",
    tags=["A2A Protocol"],
)
def send_task(task: A2ATask) -> A2ATaskResult:
    result = agent_executor.run(task)
    if result.status == "failed":
        raise HTTPException(status_code=500, detail=result.error)
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("  Critic Agent — A2A Server")
    print("  Listening on http://0.0.0.0:8002")
    print("  Agent card: http://localhost:8002/.well-known/agent.json")
    print("  Task endpoint: POST http://localhost:8002/tasks/send")
    print("=" * 60)
    uvicorn.run("new_a2a.critic.server:fastapi_app", host="0.0.0.0", port=8002, reload=False)
