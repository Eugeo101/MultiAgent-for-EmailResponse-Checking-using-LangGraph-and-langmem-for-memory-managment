# Email Respond Agent with Full Memory Architecture

## 1. Overview

This project implements a production-ready **autonomous email assistant** powered by **Gemini 3.1 Flash Lite** via GoogleAI Studio, built entirely using LangGraph. The agent classifies incoming emails into **three categories — respond, notify, or ignore —** and autonomously drafts replies, schedules meetings, and checks calendar availability using a set of custom tools.

This project also implements **custom ReAct graph** and  **four-layer memory architecture** that allows the agent to learn from experience, personalize behavior per user, and continuously improve its classification rules over time — without retraining.

---

## 2. Memory Architecture

The agent implements all four canonical types of AI agent memory, each backed by a dedicated SQLite store:

| Memory Type | Backend | Scope | Purpose |
|---|---|---|---|
| Short-term (Conversational) | `SqliteSaver` | Per `thread_id` | Maintains message history within a conversation |
| Long-term (Semantic) | `SqliteStore` | Per `langgraph_user_id` | Stores facts about contacts, actions, and context across sessions |
| Episodic (Few-shot) | `SqliteStore` | Per `langgraph_user_id` | Stores labeled email examples retrieved at inference time to guide the router |
| Procedural (System Prompts) | `SqliteStore` + LangMem | Per `langgraph_user_id` | Saves and continuously optimizes triage rules and agent instructions |

```
thread_id           ->  short-term memory  (this conversation only)
langgraph_user_id   ->  long-term memory   (all conversations, one user)
                    ->  episodic memory    (few-shot examples, one user)
                    ->  procedural memory  (optimized prompts, one user)
```

---

## 3. Workflow

### 3.1 LLM Setup

- Agent build via **Gemini 3.1 Flash-Lite** via `langchain-google-genai`.
- Configured with `temperature=0` for deterministic classification.
- API keys are loaded securely from a `.env` file using `python-dotenv`:

```env
google_api_key=your_gemini_api_key
```

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv('.env')
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.0,
    api_key=os.getenv('google_api_key')
)
llm
```

### 3.2 (Triage or Router) LLM

The first node in the graph classifies every incoming email using a structured output schema:

```python
class Router(BaseModel):
    classification: Literal["respond", "notify", "ignore"]
```

At runtime the router:

1. Queries **episodic memory** for the most semantically similar past email examples and injects them as few-shots into the prompt.
2. Loads **procedural memory** for the current triage rules (`triage_ignore`, `triage_notify`, `triage_respond`) — or seeds them from `prompt_instructions` on first run.
3. Invokes the LLM with the assembled system and user prompts.
4. Returns a `Command` that either routes to `response_agent` or terminates at `END`.

### 3.3 Response Agent

When an email is classified as `respond`, the agent:

1. Loads its system prompt from **procedural memory** (`agent_instructions`).
2. Invokes the LLM with the full message history and system prompt.
3. Decides which tools to call based on the email content.

### 3.4 Tool Executor

A manual tool executor (without `ToolNode`) loops over all `tool_calls` in the last `AIMessage`, dispatches each to the correct function, and wraps results in `ToolMessage` objects. The loop continues until the LLM produces a message with no tool calls.

Registered tools:

- `write_email(to, subject, content)` — Drafts and sends an email reply.
- `schedule_meeting(attendees, subject, duration_minutes, preferred_day)` — Schedules a calendar meeting.
- `check_calendar_availability(day)` — Checks available time slots for a given day.
- `manage_memory` — Stores relevant facts about contacts and actions in long-term semantic memory from langmem.
- `search_memory` — Retrieves relevant facts from long-term semantic memory before responding from langmem.

### 3.5 Procedural Memory for Prompt Optimization

When the router misclassifies an email, the LangMem optimizer is used to update the triage prompts directly in `procedural_memory.db`:

```python
conversations = [
    (
        response['messages'],
        "This email is clearly spam. Emails that are very short, vague, "
        "lack professional context, or come from unknown senders should "
        "always be classified as IGNORE."
    )
]
```

On the next invocation, the router automatically loads the improved rules from the database — no **manual** code changes required.

---

## 4. Agent Graph

```
START
  |
  v
[triage_router]
  |
  |-- IGNORE --> END
  |
  |-- NOTIFY --> END
  |
  |-- RESPOND --> [response_agent] <-----------+
                        |                      |
                  has tool_calls?              |
                        |                      |
                       YES --> [execute_tool] --+
                        |
                       NO --> END
```

---

## 5. Tech Stack

- Python 3.11
- LangGraph (`StateGraph`, `Command`, `SqliteSaver`, `SqliteStore`)
- LangMem (`create_manage_memory_tool`, `create_search_memory_tool`, `create_multi_prompt_optimizer`)
- Ollama (`nomic-embed-text`, 768-dimensional embeddings, fully local)
- Google Gemini (`gemini-3.1-flash-lite-preview`, optional)
- SQLite (four isolated `.db` files, one per memory layer)
- Pydantic v2 (structured output schema for router classification)
- python-dotenv (secure API key management)

---

## 6. Project Structure

```
email_respond_agent/
|
|-- agent.py                                       # EmailRespondAgent class
|-- email_task_templates.py                        # all prompts template in same script
|
|-- email_respond_agent_short_term_memory.db       # Conversation history
|-- email_respond_agent_long_term_memory.db        # Semantic memory
|-- email_respond_agent_episodic_memory.db         # Few-shot examples
|-- email_respond_agent_procedural_memory.db       # System prompts
|
|-- .env                                           # API keys
|-- README.md
```

---

## 7. Key Design Decisions

### 7.1 Four-Layer Memory Without External Vector Databases

All memory layers use SQLite via LangGraph's native `SqliteStore` and `SqliteSaver`, with local Ollama embeddings. This eliminates external service dependencies (no Pinecone, no Chroma, no OpenAI Embeddings API) while maintaining full semantic search capability.

### 7.2 Manual Tool Executor

`ToolNode` was intentionally avoided. The manual executor gives full control over error handling, logging, and tool dispatch — and removes a dependency on LangGraph internals that can change across versions.

### 7.3 Procedural Memory as a Manual Living Prompt Store

System prompts are not hardcoded. On first run they are seeded from `prompt_instructions` and written to the database. On subsequent runs they are loaded from the database. When the LangMem optimizer runs, it overwrites the stored prompts with improved versions. The agent improves without any code changes.

### 7.4 Node Methods Never Write to Instance State

Node methods are called across multiple users and threads sharing the same class instance. Writing to `self` inside a node would cause state to bleed across concurrent calls. All per-call state is passed through `AgentState` only.

### 7.5 isolation_level=None on SQLite Connections

SQLite's default transaction handling causes `OperationalError: cannot start a transaction within a transaction` when LangGraph's store attempts nested writes. Setting `isolation_level=None` puts the connection in autocommit mode, which resolves this without requiring a connection pool.

### 7.6 Command for Triage Routing

The triage router returns a `Command(goto, update)` rather than using `add_conditional_edges`. This allows the router to both route to the next node and update the graph state — injecting the email content into `messages` — in a single atomic return, rather than separating routing logic from state updates.

---

## 8. Conclusion

This project demonstrates a production-aligned email automation agent that goes beyond standard ReAct patterns by implementing a complete memory system. The agent classifies emails accurately using few-shot episodic retrieval, responds using context from long-term semantic memory, and continuously improves its own prompts through procedural memory optimization — all without external vector databases, all persisted locally, and all scoped correctly per user.
