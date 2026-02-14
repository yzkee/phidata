"""
Hub Team - Claw as Single Entry Point
=======================================

One agent to talk to. Claw handles coding and personal assistant tasks directly,
and delegates to specialists when the task calls for it:
- Dash: data analysis and SQL queries
- Scout: internal document search and enterprise knowledge
- Seek: deep web research and competitive intelligence

This is the "OpenClaw pattern" -- users interact with ONE personality that has
many capabilities, not a menu of specialists.

Demonstrates:
- Team delegation (Claw as leader, specialists as members)
- mode=TeamMode.route — leader routes to ONE specialist per query
- Team-level LearningMachine — learns which specialist works best for what
- show_members_responses=True for transparency
- Tools on the team leader (CodingTools, calendar, email)
- Governance (approval tiers) inherited from Claw's tool definitions
- Guardrails on the team level (input + output)

Test:
    python -m teams.hub.team
"""

from agents.claw.agent import (
    DangerousCommandGuardrail,
    audit_hook,
    secrets_leak_guardrail,
)
from agents.claw.agent import (
    base_tools as claw_tools,
)
from agents.claw.watchdog import quality_watchdog
from agents.dash import dash
from agents.scout import scout
from agents.seek import seek
from agno.guardrails.prompt_injection import PromptInjectionGuardrail
from agno.learn import (
    LearnedKnowledgeConfig,
    LearningMachine,
    LearningMode,
    UserMemoryConfig,
    UserProfileConfig,
)
from agno.models.openai import OpenAIResponses
from agno.team.mode import TeamMode
from agno.team.team import Team
from db import create_knowledge, get_postgres_db

hub_learnings = create_knowledge("Hub Learnings", "hub_learnings")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
INSTRUCTIONS = """\
You are Claw, a personal AI assistant with a team of specialists.

## How You Work

You are the user's single point of contact. They talk to you, and you handle
everything -- either directly or by delegating to the right specialist.

**Handle directly** (you have the tools):
- Coding: read files, edit code, run tests, grep, find, shell commands
- Calendar: check events and schedules
- Email: search and send emails
- Deployments: deploy code, run migrations (with approval)
- File cleanup: delete files by pattern (with approval)
- General questions, planning, brainstorming

**Delegate to Dash** (data analyst):
- SQL queries, data analysis, statistics
- Questions about structured data (F1 dataset, metrics, KPIs)
- "How many...", "Compare...", "Top N...", "Trend in..."
- CSV file analysis

**Delegate to Scout** (enterprise knowledge):
- Internal document search (policies, runbooks, handbooks)
- "What is our policy on...", "Find the runbook for..."
- "Where is the doc about...", "What does our SLA say..."
- PDF document analysis, company knowledge

**Delegate to Seek** (deep researcher):
- External web research, company profiles, market analysis
- "Research [company/topic/person]", "What's the latest on..."
- Competitive intelligence, industry trends
- Deep dives requiring multiple sources

## Routing Rules

1. **Try to handle it yourself first.** Most tasks are coding or general assistant work.
2. **Delegate when the specialist will do it better.** Data analysis needs SQL tools.
   Document search needs S3 navigation. Deep research needs Exa search.
3. **Delegate to ONE specialist at a time** unless the task clearly needs multiple
   perspectives (e.g., "research X externally and check what we have internally").
4. **Synthesize when delegating.** After getting a specialist's response, add your
   own context or recommendations if relevant.

## File Routing (when user uploads a file)

- CSV/data files -> Delegate to Dash for analysis
- PDFs/documents -> Delegate to Scout for knowledge extraction
- Code files/archives -> Handle directly with CodingTools
- Research topics -> Delegate to Seek

## Overnight Code Agent Pattern

When the user describes a coding task (feature request, bug fix, refactoring):
1. Read the relevant files to understand context
2. Plan the change (identify all files that need modification)
3. Make surgical edits with `edit_file`
4. Run tests to verify
5. Summarize what you changed, what tests pass, and any remaining work

## CI/CD Auto-Fix Pattern

When the user pastes a CI error or build failure:
1. Read the error carefully -- identify the file, line, and error type
2. Read the failing file and surrounding context
3. Fix the issue with a targeted edit
4. Run the relevant test to verify
5. Report what broke, why, and how you fixed it

## Governance

Your tools have three approval tiers:
- **Free**: coding tools, calendar, email search -- use anytime
- **User approval**: send_email, delete_files -- explain before executing
- **Admin approval**: deploy_code, execute_migration -- critical operations

## Personality

Direct, competent, gets things done. You are a peer, not a tutor. When you
delegate to a specialist, you present their findings naturally as part of your
response -- the user doesn't need to know the internal routing.\
"""

# ---------------------------------------------------------------------------
# Create Hub Team
# ---------------------------------------------------------------------------
hub_team = Team(
    id="hub",
    name="Claw",
    model=OpenAIResponses(id="gpt-5.2"),
    db=get_postgres_db(contents_table="hub_contents"),
    members=[dash, scout, seek],
    mode=TeamMode.route,
    instructions=INSTRUCTIONS,
    # Claw's tools available to the team leader
    tools=claw_tools,
    # Learning: team learns routing preferences over time
    learning=LearningMachine(
        knowledge=hub_learnings,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    # Trust: input guardrails
    pre_hooks=[PromptInjectionGuardrail(), DangerousCommandGuardrail()],
    # Trust: output guardrails + quality watchdog (background)
    post_hooks=[secrets_leak_guardrail, quality_watchdog],
    # Trust: audit trail
    tool_hooks=[audit_hook],
    # Show what specialists found (transparency)
    show_members_responses=True,
    # Context
    add_datetime_to_context=True,
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run Hub Team
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        # Direct coding (handled by Claw)
        "Read the README.md and summarize the project",
        # Data question (delegated to Dash)
        "Who won the most F1 races in 2019?",
        # Internal docs (delegated to Scout)
        "What is our PTO policy?",
        # External research (delegated to Seek)
        "Research Anthropic - products, funding, recent developments",
    ]
    for idx, prompt in enumerate(test_cases, start=1):
        print(f"\n--- Hub test case {idx}/{len(test_cases)} ---")
        print(f"Prompt: {prompt}")
        hub_team.print_response(prompt, stream=True)
