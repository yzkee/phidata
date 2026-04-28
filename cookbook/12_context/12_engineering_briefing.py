"""
Engineering briefing: Slack + Workspace + Parallel Web
======================================================

Three sources, one agent.

  Slack      what the team is talking about right now
  Workspace  what this repo already knows about it
  Web        what the current external reference looks like

The main agent does synthesis. Each provider owns its own mess.

Requires:
    OPENAI_API_KEY
    PARALLEL_API_KEY   https://platform.parallel.ai/
    SLACK_BOT_TOKEN    scopes: channels:read, channels:history,
                       users:read, chat:write
    pip install parallel-web
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agno.agent import Agent
from agno.context.slack import SlackContextProvider
from agno.context.web import ParallelBackend, WebContextProvider
from agno.context.workspace import WorkspaceContextProvider
from agno.models.openai import OpenAIResponses

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Sub-agents do source-specific tool work with a smaller model.
provider_model = OpenAIResponses(id="gpt-5.4-mini")

slack = SlackContextProvider(model=provider_model)
codebase = WorkspaceContextProvider(id="agno", root=PROJECT_ROOT, model=provider_model)
web = WebContextProvider(backend=ParallelBackend(), model=provider_model)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.4"),
    tools=[*slack.get_tools(), *codebase.get_tools(), *web.get_tools()],
    markdown=True,
)


if __name__ == "__main__":
    print(f"slack    = {slack.status()}")
    print(f"codebase = {codebase.status()}")
    print(f"web      = {web.status()}\n")

    prompt = (
        "Pull the 10 most recent messages from #agents, pick 2 active topics, "
        "connect each to local codebase context, and find one current external "
        "reference for each. Output a markdown table with columns: Topic, Slack "
        "signal, Codebase context, External reference, Sync question. Then post "
        "the table in #test-agents."
    )
    print(f"> {prompt}\n")
    asyncio.run(agent.aprint_response(prompt))
