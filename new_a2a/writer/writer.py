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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def writer_node(state: AgentState):
    """Generates or refines text based on the topic and any critic feedback."""
    print("--- EXECUTION: Writer Agent ---")

    sys_msg = SystemMessage(
        content=(
            "You are a talented creative writer.\n\n"
            "Your task is to write or revise a high-quality piece based on the topic "
            "and any critic feedback provided in the conversation.\n\n"
            "Rules:\n"
            "- On the first request, write a complete, well-structured piece.\n"
            "- If critic feedback is provided, revise the previous draft and address "
            "EVERY feedback point carefully.\n"
            "- Do not ignore any critique.\n"
            "- Avoid repeating mistakes or phrases that were criticized.\n"
            "- Improve clarity, imagery, rhythm, and originality where needed.\n\n"
            "Output rules:\n"
            "- Return ONLY the final revised piece.\n"
            "- Do NOT include explanations, comments, or meta text.\n"
            "- Do NOT mention the critic or the revision process."
        )
    )

    messages = [sys_msg] + state["messages"]
    response = llm.invoke(messages)

    return {
        "messages": [AIMessage(content=response.content, name="WriterAgent")]
    }

graph = StateGraph(AgentState)
graph.add_node("WriterAgent", writer_node)
graph.add_edge(START, "WriterAgent")
graph.add_edge("WriterAgent", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Writer Agent Online. (Type 'quit' to stop)")
    config = {"configurable": {"thread_id": "writer_test_1"}}
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        final_state = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        last_message = final_state["messages"][-1]
        print(f"\n[{last_message.name}]:\n{last_message.content}\n")
