from langchain_core.messages import HumanMessage

# Import the compiled LangGraph app from critic.py
from new_a2a.critic.critic import app
from new_a2a.a2a_models import A2ATask, A2ATaskResult, A2AMessage


def run(task: A2ATask) -> A2ATaskResult:
    """Execute the Critic LangGraph app and return an A2ATaskResult."""

    # Convert all A2A user messages → HumanMessages
    human_messages = [
        HumanMessage(content=msg.content)
        for msg in task.messages
        if msg.role == "user"
    ]

    if not human_messages:
        return A2ATaskResult(
            id=task.id,
            status="failed",
            error="No user messages found in the task.",
        )

    # Use task.id as thread_id so memory stays consistent across iterations
    config = {"configurable": {"thread_id": task.id}}

    try:
        final_state = app.invoke({"messages": human_messages}, config=config)

        last_msg = final_state["messages"][-1]
        agent_name = getattr(last_msg, "name", "CriticAgent")
        reply_content = f"[{agent_name}]:\n\n{last_msg.content}"

        return A2ATaskResult(
            id=task.id,
            status="completed",
            output=[A2AMessage(role="agent", content=reply_content)],
        )

    except Exception as exc:
        return A2ATaskResult(
            id=task.id,
            status="failed",
            error=str(exc),
        )
