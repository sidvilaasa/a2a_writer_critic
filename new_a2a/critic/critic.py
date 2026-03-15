from dotenv import load_dotenv
load_dotenv()

import operator
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

def critic_node(state: AgentState):
    """Evaluates the writer's text and returns actionable feedback or signals completion."""
    print("--- EXECUTION: Critic Agent ---")

    sys_msg = SystemMessage(
        content=(
            """You are a writing critic.

Evaluate the text and provide numbered actionable feedback.

IMPORTANT RULES:
- If improvements are needed, return ONLY the numbered feedback.
- If the text would score 9/10 or higher in quality and no meaningful improvements remain,
  return EXACTLY this phrase and nothing else:

NO_FURTHER_FEEDBACK"""
        )
    )

    messages = [sys_msg] + state["messages"]
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content, name="CriticAgent")]
    }

graph = StateGraph(AgentState)
graph.add_node("CriticAgent", critic_node)
graph.add_edge(START, "CriticAgent")
graph.add_edge("CriticAgent", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Critic Agent Online. (Type 'quit' to stop)")
    config = {"configurable": {"thread_id": "critic_test_1"}}
    while True:
        user_input = input("\nPaste text to evaluate: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        final_state = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        last_message = final_state["messages"][-1]
        print(f"\n[{last_message.name}]:\n{last_message.content}\n")
