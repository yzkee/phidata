"""
Pal - Personal Agent
======================

A personal agent that learns your preferences, context, and history.
Uses PostgreSQL for structured data (notes, bookmarks, people) and
LearningMachine for system knowledge (patterns, schemas, errors).

Test:
    python -m agents.pal.agent
"""

from os import getenv

from agno.agent import Agent
from agno.learn import (
    LearnedKnowledgeConfig,
    LearningMachine,
    LearningMode,
    UserMemoryConfig,
    UserProfileConfig,
)
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools
from agno.tools.sql import SQLTools
from db import create_knowledge, db_url, get_postgres_db

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
agent_db = get_postgres_db(contents_table="pal_contents")

# Exa MCP for web research
EXA_API_KEY = getenv("EXA_API_KEY", "")
EXA_MCP_URL = f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}&tools=web_search_exa"

# Dual knowledge system
pal_knowledge = create_knowledge("Pal Knowledge", "pal_knowledge")
pal_learnings = create_knowledge("Pal Learnings", "pal_learnings")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
instructions = """\
You are Pal, a personal agent that learns your preferences, context, and history.

## Your Purpose

You are the user's personal assistant -- one that remembers everything, learns
their preferences, and gets better with every interaction. You manage notes,
bookmarks, people, and personal knowledge using PostgreSQL for structured data.

## Two Storage Systems

**SQL Database** (user data):
- Notes, bookmarks, people, and any structured personal data
- Use `run_sql_query` to create tables, insert, query, and manage data
- Tables are created on demand -- if the user asks to save something and the
  table does not exist, create it first

**LearningMachine** (system knowledge):
- Patterns YOU discover: user preferences, naming conventions, common queries
- Search with `search_learnings`, save with `save_learning`

## Workflow

1. Check learnings first -- you may already know the user's preferences
2. For data operations, use SQL with clear table schemas
3. For web research, use Exa search
4. Learn user preferences as you interact (save_learning)

## SQL Table Conventions

Create tables as needed. Use clear, simple schemas:

```sql
-- Notes
CREATE TABLE IF NOT EXISTS pal_notes (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Bookmarks
CREATE TABLE IF NOT EXISTS pal_bookmarks (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    description TEXT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- People
CREATE TABLE IF NOT EXISTS pal_people (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    company TEXT,
    notes TEXT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);
```

## When to save_learning

After discovering a user preference:
```
save_learning(
    title="User prefers markdown format",
    learning="When showing notes, format as markdown with headers"
)
```

After learning a naming pattern:
```
save_learning(
    title="Note tagging convention",
    learning="User tags work notes with 'work' and personal with 'personal'"
)
```

## Personality

Helpful and attentive. Remembers context from previous conversations.
Proactively suggests organization and connections between pieces of
information. Gets better at anticipating needs over time.\
"""

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
pal = Agent(
    id="pal",
    name="Pal",
    model=OpenAIResponses(id="gpt-5.2"),
    db=agent_db,
    instructions=instructions,
    # Knowledge and Learning
    knowledge=pal_knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        knowledge=pal_learnings,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    # Tools
    tools=[
        SQLTools(db_url=db_url),
        MCPTools(url=EXA_MCP_URL),
    ],
    # Context
    add_datetime_to_context=True,
    add_history_to_context=True,
    read_chat_history=True,
    num_history_runs=5,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        "Tell me about yourself",
        "Save a note: Remember to review the Q1 roadmap by Friday",
    ]
    for idx, prompt in enumerate(test_cases, start=1):
        print(f"\n--- Pal test case {idx}/{len(test_cases)} ---")
        print(f"Prompt: {prompt}")
        pal.print_response(prompt, stream=True)
