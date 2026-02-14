"""
Agno Demo - AgentOS Entrypoint
================================

Serves all demo agents, teams, and workflows via AgentOS.
"""

from pathlib import Path

from agents.claw import claw
from agents.dash import dash
from agents.scout import scout
from agents.seek import seek
from agno.os import AgentOS
from db import get_postgres_db
from registry import registry
from teams.hub import hub_team
from teams.research import research_team
from workflows.daily_brief import daily_brief_workflow
from workflows.github_digest import github_digest_agent
from workflows.meeting_prep import meeting_prep_workflow

config_path = str(Path(__file__).parent.joinpath("config.yaml"))

agent_os = AgentOS(
    agents=[claw, dash, scout, seek, github_digest_agent],
    teams=[research_team, hub_team],
    workflows=[daily_brief_workflow, meeting_prep_workflow],
    tracing=True,
    scheduler=True,
    registry=registry,
    config=config_path,
    db=get_postgres_db(),
)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="run:app", reload=True)
