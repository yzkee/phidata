"""
Test cases for evaluating all demo agents.

Each test case targets a specific agent and checks for expected strings
in the response. Tests are organized by agent/component.
"""

from dataclasses import dataclass


@dataclass
class TestCase:
    """A test case for evaluating a demo component."""

    agent: str  # Agent/team/workflow ID
    question: str
    expected_strings: list[str]
    category: str
    match_mode: str = "all"  # "all" = all must match, "any" = at least one must match


# ---------------------------------------------------------------------------
# Agent Test Cases
# ---------------------------------------------------------------------------

CLAW_TESTS: list[TestCase] = [
    TestCase(
        agent="claw",
        question="Tell me about yourself",
        expected_strings=["Claw", "coding", "assistant"],
        category="claw_identity",
        match_mode="any",
    ),
    TestCase(
        agent="claw",
        question="List the files in the current directory",
        expected_strings=["run.py", "README"],
        category="claw_coding",
        match_mode="any",
    ),
]

DASH_TESTS: list[TestCase] = [
    TestCase(
        agent="dash",
        question="Who won the most races in 2019?",
        expected_strings=["Hamilton", "11"],
        category="dash_basic",
    ),
    TestCase(
        agent="dash",
        question="Which team won the 2020 constructors championship?",
        expected_strings=["Mercedes"],
        category="dash_basic",
    ),
    TestCase(
        agent="dash",
        question="Which driver has won the most world championships?",
        expected_strings=["Schumacher", "7"],
        category="dash_aggregation",
    ),
]

SCOUT_TESTS: list[TestCase] = [
    TestCase(
        agent="scout",
        question="What is our PTO policy?",
        expected_strings=["PTO", "days"],
        category="scout_knowledge",
        match_mode="all",
    ),
    TestCase(
        agent="scout",
        question="Find the deployment runbook",
        expected_strings=["deployment", "runbook"],
        category="scout_knowledge",
        match_mode="all",
    ),
    TestCase(
        agent="scout",
        question="What are the incident severity levels?",
        expected_strings=["severity"],
        category="scout_knowledge",
    ),
]

SEEK_TESTS: list[TestCase] = [
    TestCase(
        agent="seek",
        question="Tell me about yourself",
        expected_strings=["Seek", "research"],
        category="seek_identity",
        match_mode="any",
    ),
]

# ---------------------------------------------------------------------------
# Team Test Cases
# ---------------------------------------------------------------------------

RESEARCH_TEAM_TESTS: list[TestCase] = [
    TestCase(
        agent="research-team",
        question="Research Anthropic - what do they do and who are the key people?",
        expected_strings=["Anthropic", "Claude"],
        category="research_team",
        match_mode="any",
    ),
]

# ---------------------------------------------------------------------------
# Workflow Test Cases
# ---------------------------------------------------------------------------

DAILY_BRIEF_TESTS: list[TestCase] = [
    TestCase(
        agent="daily-brief",
        question="Generate my daily brief for today",
        expected_strings=["calendar", "email", "meeting"],
        category="daily_brief",
        match_mode="any",
    ),
]

MEETING_PREP_TESTS: list[TestCase] = [
    TestCase(
        agent="meeting-prep",
        question="Prepare me for my 10 AM Product Strategy Review meeting",
        expected_strings=["Product Strategy", "talking point"],
        category="meeting_prep",
        match_mode="any",
    ),
]

# ---------------------------------------------------------------------------
# All test cases combined
# ---------------------------------------------------------------------------

ALL_TEST_CASES: list[TestCase] = (
    CLAW_TESTS
    + DASH_TESTS
    + SCOUT_TESTS
    + SEEK_TESTS
    + RESEARCH_TEAM_TESTS
    + DAILY_BRIEF_TESTS
    + MEETING_PREP_TESTS
)

CATEGORIES = sorted(set(tc.category for tc in ALL_TEST_CASES))

# Agent-specific test collections for targeted evaluation
AGENT_TESTS: dict[str, list[TestCase]] = {
    "claw": CLAW_TESTS,
    "dash": DASH_TESTS,
    "scout": SCOUT_TESTS,
    "seek": SEEK_TESTS,
    "research-team": RESEARCH_TEAM_TESTS,
    "daily-brief": DAILY_BRIEF_TESTS,
    "meeting-prep": MEETING_PREP_TESTS,
}
