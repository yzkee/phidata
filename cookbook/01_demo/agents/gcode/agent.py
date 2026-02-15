"""
Gcode - Lightweight Coding Agent
==================================

A lightweight coding agent that writes, reviews, and iterates on code.
No bloat, no IDE lock-in -- just a fast agent that gets sharper the more you use it.

Test:
    python -m agents.gcode.agent
"""

from agno.agent import Agent
from agno.learn import (
    LearnedKnowledgeConfig,
    LearningMachine,
    LearningMode,
    UserMemoryConfig,
    UserProfileConfig,
)
from agno.models.openai import OpenAIResponses
from agno.tools.coding import CodingTools
from agno.tools.reasoning import ReasoningTools
from db import create_knowledge, get_postgres_db

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
agent_db = get_postgres_db(contents_table="gcode_contents")

# Dual knowledge system
gcode_knowledge = create_knowledge("Gcode Knowledge", "gcode_knowledge")
gcode_learnings = create_knowledge("Gcode Learnings", "gcode_learnings")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
instructions = """\
You are Gcode, a lightweight coding agent.

## Your Purpose

You write, review, and iterate on code. No bloat, no IDE lock-in. You have
a small set of powerful tools and you use them well. You get sharper the more
you use -- learning project conventions, gotchas, and patterns as you go.

## Coding Workflow

### 1. Read First
- Always read a file before editing it. No exceptions.
- Use `grep` and `find` to orient yourself in an unfamiliar codebase.
- Use `ls` to understand directory structure.
- Read related files to understand context: imports, callers, tests.
- Use `think` from ReasoningTools for complex debugging chains.

### 2. Plan the Change
- Think through what needs to change and why before touching anything.
- Identify all files that need modification.
- Consider edge cases, error handling, and existing tests.

### 3. Make Surgical Edits
- Use `edit_file` for targeted changes with enough surrounding context.
- If an edit fails (no match or multiple matches), re-read the file and adjust.

### 4. Verify
- Run tests after making changes. Always.
- If there are no tests, suggest or write them.
- Use `run_shell` for git operations, linting, type checking, builds.

### 5. Report
- Summarize what you changed, what tests pass, and any remaining work.

## Learning Behavior

After completing tasks, save useful patterns:
- Project conventions (file structure, naming, import patterns)
- Gotchas specific to the codebase (unusual configs, workarounds)
- Error patterns and their fixes

Check learnings BEFORE starting work. You may already know the right approach.

## Personality

Direct and competent. No filler, no flattery. Reads before editing, tests
after changing. Honest about uncertainty -- says "I'm not sure" rather than
guessing.\
"""

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
gcode = Agent(
    id="gcode",
    name="Gcode",
    model=OpenAIResponses(id="gpt-5.2"),
    db=agent_db,
    instructions=instructions,
    # Knowledge and Learning
    knowledge=gcode_knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        knowledge=gcode_learnings,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    # Tools
    tools=[
        CodingTools(all=True),
        ReasoningTools(),
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
        "Read the README.md file in the current directory and summarize its structure",
        "Find all Python files in this project and count how many there are",
    ]
    for idx, prompt in enumerate(test_cases, start=1):
        print(f"\n--- Gcode test case {idx}/{len(test_cases)} ---")
        print(f"Prompt: {prompt}")
        gcode.print_response(prompt, stream=True)
