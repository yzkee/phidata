"""
Engineering briefing: Slack + Workspace + Parallel Web
======================================================

Turns internal discussion into an engineering-sync briefing by composing
three context providers:

1. Slack: find the active topics in a team channel.
2. Workspace: map each topic to local code, docs, or cookbooks.
3. Parallel web: find a current external reference for each topic.

The important part is the chain. Slack decides what matters right now,
the workspace explains what this repo already knows about it, and web
research adds current external context.

Requires:
    OPENAI_API_KEY
    PARALLEL_API_KEY   (https://platform.parallel.ai/)
    SLACK_BOT_TOKEN    (or SLACK_TOKEN fallback; scopes: channels:read,
                        channels:history, users:read)
    SLACK_CHANNEL      optional channel name or ID; defaults to #agents
                       use an ID for private channels the bot cannot list
    pip install parallel-web
"""

from __future__ import annotations

import asyncio
from os import getenv
from pathlib import Path

from agno.agent import Agent
from agno.context.slack import SlackContextProvider
from agno.context.web import ParallelBackend, WebContextProvider
from agno.context.workspace import WorkspaceContextProvider
from agno.models.openai import OpenAIResponses

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLACK_CHANNEL = getenv("SLACK_CHANNEL", "#agents")

# Sub-agents do provider-specific tool work with a smaller model. The
# outer agent spends the larger model on synthesis and prioritization.
provider_model = OpenAIResponses(id="gpt-5.4-mini")

# Each provider owns its source-specific tool work behind a compact tool
# surface. The outer agent can spend its context on synthesis.
slack = SlackContextProvider(model=provider_model)

codebase = WorkspaceContextProvider(
    id="agno",
    name="Agno Codebase",
    root=PROJECT_ROOT,
    model=provider_model,
    instructions="""\
You answer questions about the local Agno repository at {root}.

Start with targeted paths before broad scans:
- AGENTS.md
- README.md
- cookbook/12_context
- libs/agno/agno/context
- libs/agno/agno/tools
- specs, if present

Dependency directories, virtualenvs, build outputs, caches, and agent
scratch folders are already excluded. Cite every claim with a relative file path.
If the repo has no clear match for a topic, say so plainly.
""",
)

web_backend = ParallelBackend()  # reads PARALLEL_API_KEY from env
web = WebContextProvider(backend=web_backend, model=provider_model)

provider_guidance = "\n".join(
    [
        slack.instructions(),
        "`Slack`: this cookbook is a briefing workflow. Do not post messages unless the user explicitly asks.",
        codebase.instructions(),
        web.instructions(),
    ]
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.4"),
    tools=[*slack.get_tools(), *codebase.get_tools(), *web.get_tools()],
    instructions=(
        "You prepare concise engineering-sync briefings by chaining context providers.\n\n"
        "Workflow:\n"
        f"1. Call `query_slack` to read the 10 most recent messages from {SLACK_CHANNEL} "
        "and identify exactly two active topics.\n"
        "2. For each topic, call `query_agno` for matching code, docs, specs, or cookbooks.\n"
        "3. For each topic, call `query_web` for one current external reference or best practice.\n"
        "4. Synthesize the three sources. Do not invent Slack facts, file paths, or URLs.\n\n"
        "Output a markdown table with columns: Topic, Slack signal, Codebase context, External reference, "
        "Sync question.\n\n" + provider_guidance
    ),
    markdown=True,
)


if __name__ == "__main__":
    print(f"slack.channel    = {SLACK_CHANNEL}")
    print(f"slack.status()    = {slack.status()}")
    print(f"codebase.status() = {codebase.status()}")
    print(f"web.status()      = {web.status()}\n")

    prompt = (
        "I'm preparing for our weekly engineering sync. Pull the 10 most recent messages "
        f"from the {SLACK_CHANNEL} Slack channel, identify 2 active topics, connect each topic to "
        "local codebase context, then find one current external reference for each topic. "
        "Finish with the best question or next action to bring into the meeting."
    )
    print(f"> {prompt}\n")
    asyncio.run(agent.aprint_response(prompt))
