# Agno Demo

4 agents, 2 teams, 2 workflows, and 1 scheduled digest served via AgentOS. Each agent learns from interactions and improves with every use.

## Overview

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **Claw** | Personal AI assistant and coding agent | Governance (3-tier approval), guardrails, audit hooks, CodingTools |
| **Dash** | Self-learning data analyst (F1 dataset) | SQL tools, semantic model, business context, LearningMachine |
| **Scout** | Enterprise knowledge navigator (S3 docs) | S3 connector, intent routing, source registry, LearningMachine |
| **Seek** | Deep research agent | MCP tools (Exa), 4-phase methodology, ParallelTools |

## Architecture

All agents share a common foundation:

- **Model**: `OpenAIResponses(id="gpt-5.2")`
- **Storage**: PostgreSQL + PgVector for knowledge, learnings, and chat history
- **Knowledge**: Dual knowledge system -- static curated knowledge + dynamic learnings discovered at runtime
- **Search**: Hybrid search (semantic + keyword) with OpenAI embeddings (`text-embedding-3-small`)
- **Learning**: `LearningMachine` in `AGENTIC` mode -- agents decide when to save learnings

## Agents

### Claw - Personal AI Assistant and Coding Agent

Demonstrates the three pillars of the agentic contract:

- **Governance**: Three-tier tool approval -- free (read files, check calendar), user-confirmation (send email, delete files), and admin (deploy code, run migrations)
- **Trust**: Input guardrails (prompt injection detection, dangerous command blocking), output guardrails (secrets leak detection), and audit hooks on all tool calls
- **Tools**: CodingTools (read, edit, write, shell, grep, find, ls), calendar, email, deployments, migrations

### Dash - Self-Learning Data Agent

Analyzes an F1 racing dataset via SQL. Provides insights and context, not just raw query results. Remembers column quirks, date formats, and successful queries across sessions.

### Scout - Self-Learning Context Agent

Finds information across company S3 storage using grep-like search and full document reads. Knows what sources exist and routes queries to the right bucket. Learns intent from repeated use. Also serves as a support knowledge agent for internal FAQs.

### Seek - Deep Research Agent

Conducts exhaustive multi-source research and produces structured, well-sourced reports. Follows a 4-phase methodology: scope, gather, analyze, synthesize.

## Teams

| Team | Members | Mode | Purpose |
|------|---------|------|---------|
| **Hub Team (Claw)** | Dash + Scout + Seek | Route | Single entry point -- Claw handles coding directly and delegates data questions to Dash, doc questions to Scout, research to Seek. Uses `TeamMode.route`, team-level `LearningMachine`, guardrails, audit hooks, and background quality watchdog. |
| **Research Team** | Seek + Scout | Coordinate | Breaks research into dimensions (external, internal) and delegates to specialists. Synthesizes findings into a comprehensive report. |

## Workflows

| Workflow | Steps | Purpose |
|----------|-------|---------|
| **Daily Brief** | 3 parallel gatherers (calendar, email, news) then 1 synthesizer | Morning briefing with priorities, schedule highlights, inbox summary, and industry news. Uses mock calendar/email data and live web search for news. |
| **Meeting Prep** | Parse meeting, then 3 parallel researchers (attendees, internal docs, external context), then 1 synthesizer | Deep preparation with attendee context, key data points, talking points, and anticipated questions. Uses mock meeting data and live web search. |
| **GitHub Digest** | Single agent with GithubTools + SlackTools | Proactive digest of GitHub activity (PRs, issues, commits). Designed for AgentOS scheduler. Requires `GITHUB_ACCESS_TOKEN`. |

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/agno-agi/agno.git
cd agno
```

### 2. Create and activate the demo virtual environment
```bash
./scripts/demo_setup.sh
source .venvs/demo/bin/activate
```

### 3. Run PgVector
```bash
./cookbook/scripts/run_pgvector.sh
```

### 4. Export environment variables
```bash
export OPENAI_API_KEY="..."      # Required for all agents
export EXA_API_KEY="..."         # Optional (Exa MCP is currently free)
export GITHUB_ACCESS_TOKEN="..." # Optional, for GitHub Digest agent
```

### 5. Load data and knowledge
```bash
cd cookbook/01_demo

python -m agents.dash.scripts.load_data
python -m agents.dash.scripts.load_knowledge
python -m agents.scout.scripts.load_knowledge
```

### 6. Run the demo
```bash
python -m run
```

### 7. Connect via AgentOS

- Open [os.agno.com](https://os.agno.com) in your browser
- Click "Add AgentOS"
- Add `http://localhost:7777` as an endpoint
- Click "Connect"

## Evals

Test cases covering all agents, teams, and workflows. Uses string-matching validation with `all` or `any` match modes.

```bash
# Run all evals
python -m evals.run_evals

# Filter by agent
python -m evals.run_evals --agent dash
python -m evals.run_evals --agent claw

# Filter by category
python -m evals.run_evals --category dash_basic

# Verbose mode (show full responses on failure)
python -m evals.run_evals --verbose
```

## Agno Features Demonstrated

| Feature | Where |
|---------|-------|
| LearningMachine (AGENTIC mode) | All 4 agents |
| Governance / Approval | Claw (3-tier: free, user, admin) |
| Input Guardrails | Claw (prompt injection, dangerous commands) |
| Output Guardrails | Claw (secrets leak detection) |
| Audit Hooks | Claw (tool call logging) |
| CodingTools | Claw |
| SQL Tools | Dash |
| MCP Tools | Seek (Exa), Scout (Exa), Dash (Exa) |
| ParallelTools | Seek, workflows |
| Knowledge (hybrid search) | All agents |
| Teams (route mode) | Hub Team |
| Teams (coordinate mode) | Research Team |
| Workflows (parallel steps) | Daily Brief, Meeting Prep |
| Scheduled Tasks | GitHub Digest |
| AgentOS | run.py |
