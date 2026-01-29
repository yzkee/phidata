<div align="center" id="top">
  <a href="https://agno.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg">
      <img src="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg" alt="Agno">
    </picture>
  </a>
</div>

<p align="center">
  Build agents that learn.
</p>

<div align="center">
  <a href="https://docs.agno.com">Docs</a>
  <span>&nbsp;•&nbsp;</span>
  <a href="https://github.com/agno-agi/agno/tree/main/cookbook">Cookbook</a>
  <span>&nbsp;•&nbsp;</span>
  <a href="https://community.agno.com/">Community</a>
  <span>&nbsp;•&nbsp;</span>
  <a href="https://www.agno.com/discord">Discord</a>
</div>

## What is Agno?

**A framework for building multi-agent systems that learn and improve with every interaction.**

Most agents are stateless. They reason, respond, forget. Session history helps, but they're exactly as capable on day 1000 as they were on day 1.

Agno agents are different. They remember users across sessions, accumulate knowledge across conversations, and learn from decisions. Insights from one user benefit everyone. The system gets smarter over time.

Everything runs in your cloud. Your data never leaves your environment.

## Quick Example

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agents.db"),
    learning=True,
)
```

One line. Your agent now remembers users, accumulates knowledge, and improves over time.

## Production Stack

Agno provides the complete infrastructure for building multi-agent systems that learn:

| Layer | What it does |
|-------|--------------|
| **Framework** | Build agents with learning, tools, knowledge, and guardrails |
| **Runtime** | Run in production using [AgentOS](https://docs.agno.com/agent-os/introduction) |
| **Control Plane** | Monitor and manage via the [AgentOS UI](https://os.agno.com) |

## Learn More

- [Documentation](https://docs.agno.com)
- [Cookbook](https://github.com/agno-agi/agno/tree/main/cookbook)
- [AgentOS Railway Template](https://github.com/agno-agi/agentos-railway-template)

## Getting Started

**1. Build your first agent**

```bash
pip install -U agno openai
export OPENAI_API_KEY=sk-***
```

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(model=OpenAIResponses(id="gpt-5.2"))
agent.print_response("What is Agno?", stream=True)
```

**2. Add learning**

```python
from agno.db.sqlite import SqliteDb

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=SqliteDb(db_file="tmp/agents.db"),
    learning=True,
)
```

Now your agent remembers users across sessions and accumulates knowledge over time.

**3. Deploy a multi-agent system**

Clone the [AgentOS Railway template](https://github.com/agno-agi/agentos-railway-template) to run a production multi-agent system with Docker locally, then deploy to Railway:

```bash
git clone https://github.com/agno-agi/agentos-railway-template.git agentos-railway
cd agentos-railway
cp example.env .env  # Add your OPENAI_API_KEY
docker compose up -d --build
```

Connect to the control plane at [os.agno.com](https://os.agno.com) → Add OS → Local → `http://localhost:8000`

## Features

**Learning**
- User profiles that persist across sessions
- User memories that accumulate over time
- Learned knowledge that transfers across users
- Always or agentic learning modes

**Core**
- Model-agnostic: OpenAI, Anthropic, Google, local models
- Type-safe I/O with `input_schema` and `output_schema`
- Async-first, built for long-running tasks
- Natively multimodal (text, images, audio, video, files)

**Knowledge**
- Agentic RAG with 20+ vector stores, hybrid search, reranking
- Persistent storage for session history and state

**Orchestration**
- Human-in-the-loop (confirmations, approvals, overrides)
- Guardrails for validation and security
- First-class MCP and A2A support
- 100+ built-in toolkits

**Production**
- Ready-to-use FastAPI runtime
- Integrated control plane UI
- Evals for accuracy, performance, latency

## IDE Integration

Add our docs to your AI-enabled editor:

**Cursor:** Settings → Indexing & Docs → Add `https://docs.agno.com/llms-full.txt`

Also works with VSCode, Windsurf, and similar tools.

## Contributing

See the [contributing guide](https://github.com/agno-agi/agno/blob/main/CONTRIBUTING.md).

## Telemetry

Agno logs which model providers are used to prioritize updates. Disable with `AGNO_TELEMETRY=false`.

<p align="right"><a href="#top">↑ Back to top</a></p>
