"""
Gcode - Lightweight Coding Agent
==================================

A lightweight coding agent that writes, reviews, and iterates on code.
No bloat, no IDE -- just a fast agent that gets sharper the more you use it.

Gcode operates in a sandboxed workspace directory (agents/gcode/workspace/).
All file operations are restricted to this directory via CodingTools(base_dir=...).

Test:
    python -m agents.gcode.agent
"""

from pathlib import Path

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
agent_db = get_postgres_db()
WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

# Dual knowledge system
gcode_knowledge = create_knowledge("Gcode Knowledge", "gcode_knowledge")
gcode_learnings = create_knowledge("Gcode Learnings", "gcode_learnings")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
instructions = """\
You are Gcode, a lightweight coding agent.

## Your Purpose

You write, review, and iterate on code. No bloat, no IDE. You have
a small set of powerful tools and you use them well. You get sharper the more
you use -- learning project conventions, gotchas, and patterns as you go.

You operate in a sandboxed workspace directory. All files you create, read,
and edit live there. Use relative paths (e.g. "app.py", "src/main.py").

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

## Workspace

Your workspace is a directory where all your projects live. Each task gets
its own project folder:

- When starting a new task, create a descriptively named subdirectory
  (e.g. "todo-app/", "math-parser/", "url-shortener/")
- All files for that task go inside its project folder
- Use `ls` to see existing projects before creating a new one
- If the user asks to continue working on something, find the existing
  project folder first

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
    knowledge=gcode_knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        knowledge=gcode_learnings,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    tools=[CodingTools(base_dir=WORKSPACE, all=True), ReasoningTools()],
    add_datetime_to_context=True,
    add_history_to_context=True,
    read_chat_history=True,
    num_history_runs=20,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gcode.print_response(
        "Build a Python script that generates multiplication tables (1-12) "
        "as a formatted text file, then read and display it.",
        stream=True,
    )
