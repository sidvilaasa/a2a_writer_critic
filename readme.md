# ✍️ Writer–Critic A2A: Collaborative Writing with Agent-to-Agent Protocol

A multi-agent system where a **Writer Agent** and a **Critic Agent** collaborate iteratively to produce high-quality text — powered by [Google Gemini](https://ai.google.dev/), [LangGraph](https://github.com/langchain-ai/langgraph), and the [A2A (Agent-to-Agent) Protocol](https://github.com/google/A2A).

## Overview

The system consists of three components:

| Component | Description | Port |
|-----------|-------------|------|
| **Writer Agent** | A LangGraph agent that generates and refines text on any topic | `8001` |
| **Critic Agent** | A LangGraph agent that evaluates text and provides actionable feedback | `8002` |
| **Orchestrator** | A CLI client that drives the Writer ↔ Critic feedback loop | — |

### How It Works

1. The **Orchestrator** discovers both agents via their A2A Agent Cards (`/.well-known/agent.json`).
2. A user provides a **writing topic**.
3. The **Writer Agent** produces an initial draft.
4. The **Critic Agent** evaluates the draft and returns numbered feedback.
5. The writer revises based on the feedback.
6. Steps 4–5 repeat until the critic responds with `NO_FURTHER_FEEDBACK` (score ≥ 8.5/10) or the maximum iteration limit (10) is reached.

## Project Structure

```
new_a2a/
├── a2a_models.py          # Shared Pydantic models (AgentCard, A2ATask, A2AMessage, etc.)
├── orchestrator.py         # CLI orchestrator that drives the writer-critic loop
├── writer/
│   ├── writer.py           # LangGraph Writer Agent (Gemini 2.0 Flash)
│   ├── agent_executor.py   # Bridges A2A task format ↔ LangGraph invocation
│   └── server.py           # FastAPI A2A server for the Writer Agent
├── critic/
│   ├── critic.py           # LangGraph Critic Agent (Gemini 2.0 Flash)
│   ├── agent_executor.py   # Bridges A2A task format ↔ LangGraph invocation
│   └── server.py           # FastAPI A2A server for the Critic Agent
├── requirements.txt
└── readme.md
```

## Prerequisites

- **Python 3.10+**
- A **Google Gemini API key** — set as an environment variable:
  ```bash
  export GOOGLE_API_KEY="your-api-key-here"
  ```
  Or add it to a `.env` file in the project root.

## Installation

1. **Clone the repository** and navigate to the project folder:
   ```bash
   cd a2a_writer_critic
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r new_a2a/requirements.txt
   ```

## Usage

You need **three terminals** — one for each agent server and one for the orchestrator.

### 1. Start the Writer Agent

```bash
python -m new_a2a.writer.server
```

The writer agent will be available at `http://localhost:8001`.

### 2. Start the Critic Agent

```bash
python -m new_a2a.critic.server
```

The critic agent will be available at `http://localhost:8002`.

### 3. Run the Orchestrator

```bash
python -m new_a2a.orchestrator
```

You'll be prompted to enter a writing topic. The orchestrator will then run the writer–critic loop and display each iteration until the critic is satisfied or the max iterations are reached.

Type `quit` to exit.

## A2A Protocol Endpoints

Both agent servers expose the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/.well-known/agent.json` | Returns the agent's discovery card |
| `POST` | `/tasks/send` | Sends a task to the agent and returns the result |
| `GET` | `/docs` | Swagger UI for interactive API documentation |

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash via `langchain-google-genai`
- **Agent Framework**: LangGraph (with in-memory checkpointing)
- **API Framework**: FastAPI + Uvicorn
- **HTTP Client**: HTTPX (for inter-agent communication)
- **Data Validation**: Pydantic v2

## License

This project is for educational and demonstration purposes.
