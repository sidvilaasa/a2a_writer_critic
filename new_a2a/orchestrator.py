import uuid
import httpx

from new_a2a.a2a_models import AgentCard, A2ATask, A2AMessage, A2ATaskResult


WRITER_URL = "http://localhost:8001"
CRITIC_URL  = "http://localhost:8002"
MAX_ITERATIONS = 10
DONE_SIGNAL = "NO_FURTHER_FEEDBACK"


def discover_agent(url: str) -> AgentCard | None:
    """Fetch the agent card from a server. Returns None if unreachable."""
    try:
        r = httpx.get(f"{url}/.well-known/agent.json", timeout=5.0)
        r.raise_for_status()
        return AgentCard.model_validate(r.json())
    except httpx.ConnectError:
        return None
    except Exception as exc:
        print(f"  ⚠️  Error reaching {url}: {exc}")
        return None


def send_task(server_url: str, task: A2ATask) -> A2ATaskResult:
    """POST a task to an agent server and return the result."""
    r = httpx.post(
        f"{server_url}/tasks/send",
        json=task.model_dump(),
        timeout=120.0,
    )
    r.raise_for_status()
    return A2ATaskResult.model_validate(r.json())


def print_divider(label: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")


def run_loop(topic: str, writer_card: AgentCard, critic_card: AgentCard) -> None:
    session_id = str(uuid.uuid4())

    current_text: str = ""
    feedback: str = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print_divider(f"ITERATION {iteration}/{MAX_ITERATIONS}")

        # --- Writer turn ---
        if iteration == 1:
            writer_prompt = f"Write about the following topic:\n\n{topic}"
        else:
            writer_prompt = (
                f"Original topic: {topic}\n\n"
                f"Your previous draft:\n{current_text}\n\n"
                f"Critic feedback to address:\n{feedback}\n\n"
                "Rewrite the piece, incorporating ALL the feedback above."
            )

        print(f"\n✍️   Sending to {writer_card.name}...")
        writer_task = A2ATask(
            id=f"{session_id}-writer-{iteration}",
            messages=[A2AMessage(role="user", content=writer_prompt)],
        )
        writer_result = send_task(WRITER_URL, writer_task)
        current_text = writer_result.output[0].content if writer_result.output else ""

        print(f"\n📝  {writer_card.name} says:\n")
        print(current_text)

        # --- Critic turn ---
        critic_prompt = (
            f"Please evaluate the following piece of writing:\n\n{current_text}"
        )

        print(f"\n🔍  Sending to {critic_card.name}...")
        critic_task = A2ATask(
            id=f"{session_id}-critic-{iteration}",
            messages=[A2AMessage(role="user", content=critic_prompt)],
        )
        critic_result = send_task(CRITIC_URL, critic_task)
        feedback = critic_result.output[0].content if critic_result.output else ""

        print(f"\n💬  {critic_card.name} says:\n")
        print(feedback)

        # --- Check termination ──────────────────────────────────────────
        if DONE_SIGNAL in feedback.strip():
            print_divider("CRITIC IS SATISFIED — No further feedback!")
            break
    else:
        print_divider(f" Maximum iterations ({MAX_ITERATIONS}) reached.")

    print_divider("📄  FINAL APPROVED TEXT")
    print(current_text)
    print()


def main() -> None:
    print("\n" + "=" * 64)
    print("  Writer ↔ Critic Collaborative Loop  (A2A Protocol)")
    print("=" * 64)
    
    # ── Step 1: discover agents ──────────────────────────────────────────
    print("\n🔍  Discovering agents...")
    writer_card = discover_agent(WRITER_URL)
    critic_card  = discover_agent(CRITIC_URL)

    if not writer_card:
        print(f"  ❌  Writer Agent is offline ({WRITER_URL}). Start writer_server.py first.")
        return
    if not critic_card:
        print(f"  ❌  Critic Agent is offline ({CRITIC_URL}). Start critic_server.py first.")
        return

    print(f"  ✅  {writer_card.name} at {WRITER_URL}")
    print(f"  ✅  {critic_card.name}  at {CRITIC_URL}")
    
    
    print("\nenter quit to exit.")
    print("=" * 64)
    
    while True:
        
        topic = input("Enter a writing topic: ").strip()
        if(topic.lower() == "quit"):
            print("Exiting.")
            return
        if not topic:
            continue

        run_loop(topic, writer_card, critic_card)

if __name__ == "__main__":
    main()
