"""
Engineering briefing: Slack + Project Files + Parallel Web
==========================================================

Turns internal discussion into an engineering-sync briefing by composing
three context providers:

1. Slack: find the active topics in a team channel.
2. Project files: map each topic to local code, docs, or cookbooks.
3. Parallel web: find a current external reference for each topic.

The important part is the chain. Slack decides what matters right now,
project files explain what this repo already knows about it, and web
search adds current external context.

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
from agno.context import ContextMode
from agno.context.fs import FilesystemContextProvider
from agno.context.slack import SlackContextProvider
from agno.context.web import ParallelBackend, WebContextProvider
from agno.models.openai import OpenAIResponses

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLACK_CHANNEL = getenv("SLACK_CHANNEL", "#agents")

# Sub-agents do provider-specific tool work with a smaller model. The
# outer agent spends the larger model on synthesis and prioritization.
provider_model = OpenAIResponses(id="gpt-5.4-mini")

# Slack is read-only in this cookbook. ContextMode.agent exposes only
# query_slack instead of the default query_slack + update_slack surface.
# CLI runs use bot-token channel reads. Slack interface runs include an
# action_token, so the provider can use Slack assistant search.
slack = SlackContextProvider(mode=ContextMode.agent, model=provider_model)

project = FilesystemContextProvider(
    id="project",
    name="Project Files",
    root=PROJECT_ROOT,
    model=provider_model,
    instructions="""\
You answer questions about the local Agno repository at {root}.

Start with targeted paths before broad scans:
- AGENTS.md
- README.md
- cookbook/12_context
- libs/agno/agno/context
- specs, if present

Do not inspect .git, .venv, .venvs, dist, build, or cache directories
unless the user explicitly asks. Cite every claim with a relative file path.
If the repo has no clear match for a topic, say so plainly.
""",
)

web_backend = ParallelBackend()  # reads PARALLEL_API_KEY from env
web = WebContextProvider(backend=web_backend, model=provider_model)

provider_guidance = "\n".join(
    [
        "`Slack`: call `query_slack(question)` to read Slack. This cookbook is read-only; do not post messages.",
        project.instructions(),
        web.instructions(),
    ]
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.4"),
    tools=[*slack.get_tools(), *project.get_tools(), *web.get_tools()],
    instructions=(
        "You prepare concise engineering-sync briefings by chaining context providers.\n\n"
        "Workflow:\n"
        "1. Read Slack first to identify exactly two active topics.\n"
        "2. For each topic, query Project Files for matching code, docs, specs, or cookbooks.\n"
        "3. For each topic, query Web for one current external reference.\n"
        "4. Synthesize the three sources. Do not invent Slack facts, file paths, or URLs.\n\n"
        "Output a markdown table with columns: Topic, Slack signal, Project context, External reference, "
        "Sync question.\n\n" + provider_guidance
    ),
    markdown=True,
)


if __name__ == "__main__":
    print(f"slack.channel   = {SLACK_CHANNEL}")
    print(f"slack.status()   = {slack.status()}")
    print(f"project.status() = {project.status()}")
    print(f"web.status()     = {web.status()}\n")

    prompt = (
        "I'm preparing for our weekly engineering sync. Pull the 10 most recent messages "
        f"from the {SLACK_CHANNEL} Slack channel, identify 2 active topics, connect each topic to "
        "local project context, then find one current external reference for each topic. "
        "Finish with the best question or next action to bring into the meeting."
    )
    print(f"> {prompt}\n")
    asyncio.run(agent.aprint_response(prompt))
