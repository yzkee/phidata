"""
GitHub Digest - Proactive Activity Summarizer
===============================================

A scheduled agent that pulls GitHub activity (PRs, issues, commits) and posts
a formatted summary to a Slack channel. This is a PROACTIVE agent -- it
initiates contact rather than waiting for user messages.

Demonstrates:
- GithubTools + SlackTools combined
- Agent-initiated contact (proactive Slack posting)
- Designed for AgentOS scheduler (cron-based execution)
- LearningMachine to learn which repos/contributors matter most

Required env vars:
- GITHUB_ACCESS_TOKEN: GitHub PAT with read-only repo scope
- SLACK_TOKEN: Slack bot token with chat:write scope (optional for dry-run)

Test:
    python -m workflows.github_digest.workflow
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
from agno.tools.slack import SlackTools
from db import create_knowledge, get_postgres_db

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
agent_db = get_postgres_db(contents_table="github_digest_contents")
digest_learnings = create_knowledge("Digest Learnings", "digest_learnings")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
INSTRUCTIONS = """\
You are a GitHub activity digest agent. Your job is to summarize recent
repository activity and post it to Slack.

## Workflow

1. **Gather**: Pull recent PRs, issues, and repository stats from GitHub
2. **Analyze**: Identify the most important changes and trends
3. **Format**: Create a concise, scannable Slack message
4. **Post**: Send the digest to the configured Slack channel

## Digest Format

Structure your digest as:

**PRs Merged** (last 24h)
- PR title by @author -- one-line summary of the change

**PRs Open / In Review**
- PR title by @author -- what it does, any blockers

**Issues Opened**
- Issue title -- brief description, labels

**Notable Activity**
- Any significant commits, releases, or milestones

## Rules

- Lead with the most impactful changes
- Keep each item to one line
- Tag contributors with their GitHub handles
- Skip bot-generated PRs unless they are significant
- If there is no activity, say so briefly -- do not pad the digest
- Learn which repos and contributors the team cares about most

## When Run Interactively

If a user asks you a question directly (not via scheduler), answer it using
your GitHub tools. You can look up specific PRs, issues, contributors, or
repository stats on demand.
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _build_tools() -> list:
    tools: list = []
    if getenv("GITHUB_ACCESS_TOKEN"):
        from agno.tools.github import GithubTools

        tools.append(GithubTools())
    tools.append(
        SlackTools(
            enable_send_message=True,
            enable_list_channels=True,
        )
    )
    return tools


# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
github_digest_agent = Agent(
    id="github-digest",
    name="GitHub Digest",
    model=OpenAIResponses(id="gpt-5-mini"),
    db=agent_db,
    instructions=INSTRUCTIONS,
    tools=_build_tools(),
    learning=LearningMachine(
        knowledge=digest_learnings,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    add_datetime_to_context=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        "Generate a digest of recent GitHub activity for the agno-agi/agno repository",
        "What PRs were merged in agno-agi/agno this week?",
    ]
    for idx, prompt in enumerate(test_cases, start=1):
        print(f"\n--- GitHub Digest test case {idx}/{len(test_cases)} ---")
        print(f"Prompt: {prompt}")
        github_digest_agent.print_response(prompt, stream=True)
