"""
Claw - Personal AI Assistant & Coding Agent
=============================================

A personal AI assistant inspired by OpenClaw, built with the new agentic
contract: streaming, governance, and trust.

Demonstrates all three contract terms:
- Streaming: real-time tool calls and reasoning via print_response(stream=True)
- Governance: three-tier approval (free / user / admin)
- Trust: input guardrails, output guardrails, and audit hooks

Test:
    python -m agents.claw.agent
"""

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from agno.agent import Agent
from agno.approval import approval
from agno.exceptions import CheckTrigger, InputCheckError, OutputCheckError
from agno.guardrails.base import BaseGuardrail
from agno.guardrails.prompt_injection import PromptInjectionGuardrail
from agno.learn import (
    LearnedKnowledgeConfig,
    LearningMachine,
    LearningMode,
    UserMemoryConfig,
    UserProfileConfig,
)
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunInput, RunOutput
from agno.run.team import TeamRunInput
from agno.tools import tool
from agno.tools.coding import CodingTools
from agno.tools.reasoning import ReasoningTools
from agno.utils.log import logger
from db import create_knowledge, get_postgres_db

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
agent_db = get_postgres_db(contents_table="claw_contents")

# Dual knowledge system
claw_knowledge = create_knowledge("Claw Knowledge", "claw_knowledge")
claw_learnings = create_knowledge("Claw Learnings", "claw_learnings")

# ===========================================================================
# TRUST: Input Guardrails (pre_hooks)
# ===========================================================================


class DangerousCommandGuardrail(BaseGuardrail):
    """Blocks requests containing destructive system commands."""

    DANGEROUS_PATTERNS = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.",
        "> /dev/sda",
        "dd if=/dev/zero",
        ":(){ :|:&",
        "chmod -R 777 /",
        "drop table",
        "drop database",
        "--no-preserve-root",
        "format c:",
    ]

    def check(self, run_input: Union[RunInput, TeamRunInput]) -> None:
        content = run_input.input_content_string().lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in content:
                raise InputCheckError(
                    f"Blocked: request contains a destructive command pattern ({pattern}).",
                    check_trigger=CheckTrigger.INPUT_NOT_ALLOWED,
                )

    async def async_check(self, run_input: Union[RunInput, TeamRunInput]) -> None:
        self.check(run_input)


# ===========================================================================
# TRUST: Output Guardrails (post_hooks)
# ===========================================================================

# Patterns that indicate leaked secrets in agent output
_SECRET_PATTERNS = [
    (r"(?:AKIA|ASIA)[A-Z0-9]{16}", "AWS Access Key"),
    (r"sk-[a-zA-Z0-9]{20,}", "API Secret Key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
    (r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----", "Private Key"),
    (
        r"(?i)password\s*[:=]\s*['\"][^'\"]{8,}['\"]",
        "Hardcoded Password",
    ),
    (
        r"(?i)(?:api_key|apikey|secret_key|access_token)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
        "Hardcoded Secret",
    ),
]


def secrets_leak_guardrail(run_output: RunOutput) -> None:
    """Post-hook: block responses that contain leaked secrets or credentials."""
    if not run_output.content:
        return

    for pattern, label in _SECRET_PATTERNS:
        if re.search(pattern, run_output.content):
            raise OutputCheckError(
                f"Response blocked: contains a {label}. "
                "Redact credentials before including them in responses.",
                check_trigger=CheckTrigger.OUTPUT_NOT_ALLOWED,
            )


# ===========================================================================
# TRUST: Audit Hook (tool_hooks)
# ===========================================================================


def audit_hook(
    function_name: str, function_call: Callable, arguments: Dict[str, Any]
) -> Any:
    """Log all tool calls for audit trail."""
    logger.info(f"[Claw Audit] Tool: {function_name} | Args: {arguments}")
    result = function_call(**arguments)
    logger.info(f"[Claw Audit] Tool: {function_name} | Complete")
    return result


# ===========================================================================
# GOVERNANCE: Free Tools (no approval needed)
# ===========================================================================


@tool
def check_calendar(
    date: Optional[str] = None,
    days_ahead: int = 1,
) -> str:
    """Check calendar events for a given date or date range.

    Use this to look up what meetings and events are scheduled. This tool
    is free to use -- no approval required.

    Args:
        date: Date to check in YYYY-MM-DD format. Defaults to today.
        days_ahead: Number of days to look ahead (default: 1).

    Returns:
        str: Formatted list of calendar events.
    """
    start = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
    end = start + timedelta(days=days_ahead)
    return (
        f"Calendar: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}\n"
        "No events found. (Connect Google Calendar API for live data)"
    )


@tool
def search_emails(
    query: str,
    max_results: int = 10,
) -> str:
    """Search emails by keyword, sender, or subject.

    Use this to look up past emails and conversations. This tool is free
    to use -- no approval required.

    Args:
        query: Search query (keyword, sender name, or subject).
        max_results: Maximum number of results to return (default: 10).

    Returns:
        str: Matching emails with sender, subject, and date.
    """
    return (
        f"Email search: '{query}' (max {max_results})\n"
        "No results found. (Connect Gmail API for live data)"
    )


# ===========================================================================
# GOVERNANCE: User-Approval Tools (pause for user confirmation)
# ===========================================================================


@approval
@tool(requires_confirmation=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. Requires user approval before sending.

    The agent will pause and present the email for review. The email is only
    sent after the user explicitly approves.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.

    Returns:
        str: Confirmation of sent email or error message.
    """
    logger.info(f"Sending email to {to}: {subject}")
    return (
        f"Email sent to {to}\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        "(Connect SMTP/Gmail API for live delivery)"
    )


@approval
@tool(requires_confirmation=True)
def delete_files(pattern: str, directory: str = ".") -> str:
    """Delete files matching a glob pattern. Requires user approval.

    Use this for bulk file cleanup operations like removing build artifacts,
    cache files, or temporary files. The agent will pause and wait for
    approval before any files are deleted.

    Args:
        pattern: Glob pattern for files to delete (e.g. "*.pyc", "**/__pycache__").
        directory: Directory to search in (default: current directory).

    Returns:
        str: Summary of deleted files or error message.
    """
    search_dir = Path(directory).resolve()
    if not search_dir.exists():
        return f"Error: Directory not found: {directory}"

    matches = list(search_dir.glob(pattern))
    if not matches:
        return f"No files found matching pattern: {pattern}"

    deleted: List[str] = []
    errors: List[str] = []
    for match in matches:
        if match.is_file():
            try:
                match.unlink()
                deleted.append(str(match))
            except Exception as e:
                errors.append(f"{match}: {e}")

    result = f"Deleted {len(deleted)} files"
    if errors:
        result += f", {len(errors)} errors"
    if deleted:
        result += ":\n" + "\n".join(f"  - {f}" for f in deleted[:20])
        if len(deleted) > 20:
            result += f"\n  ... and {len(deleted) - 20} more"
    if errors:
        result += "\nErrors:\n" + "\n".join(f"  - {e}" for e in errors[:5])
    return result


# ===========================================================================
# GOVERNANCE: Admin-Approval Tools (critical operations)
# ===========================================================================


@approval
@tool(requires_confirmation=True)
def deploy_code(target: str, command: str) -> str:
    """Deploy code to a target environment. Requires admin approval.

    Use this for deployment operations that push code to staging or production.
    The agent will pause and wait for admin approval before running the
    deployment command.

    Args:
        target: Deployment target (e.g. "staging", "production").
        command: The deployment command to run.

    Returns:
        str: Deployment output or error message.
    """
    logger.info(f"Deploying to {target}: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return f"Deploy to {target} (exit code {result.returncode}):\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Deploy command timed out after 300 seconds"
    except Exception as e:
        return f"Error deploying to {target}: {e}"


@approval
@tool(requires_confirmation=True)
def execute_migration(migration_name: str, database: str = "default") -> str:
    """Run a database migration. Requires admin approval.

    Use this for schema changes, data migrations, and other database
    operations that modify production data. The agent will pause and wait
    for admin approval before executing.

    Args:
        migration_name: Name or path of the migration to run.
        database: Database target (default: "default").

    Returns:
        str: Migration output or error message.
    """
    logger.info(f"Running migration {migration_name} on {database}")
    try:
        result = subprocess.run(
            f"python manage.py migrate {migration_name} --database={database}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return (
            f"Migration {migration_name} on {database} "
            f"(exit code {result.returncode}):\n{output}"
        )
    except subprocess.TimeoutExpired:
        return "Error: Migration timed out after 300 seconds"
    except Exception as e:
        return f"Error running migration: {e}"


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------
instructions = """\
You are Claw, a personal AI assistant and coding agent.

## Your Purpose

You help the user with coding tasks, email, calendar, file management, and
deployments. You have a small set of powerful, composable tools and you use
them well.

You are not a tutor or a cheerleader. You are a competent peer who gets
things done and explains decisions when relevant.

## Philosophy

Genuine helpfulness over performance theater. Skip the "Great question!" and
just help. Develop real opinions about code quality and architecture. When you
see something wrong, say so.

Solve problems independently first. Read files, check context, trace through
code paths before asking for clarification. Only ask when you genuinely cannot
proceed without the answer.

## Governance: Three Levels of Tool Access

Your tools have three levels of access. This is by design -- not all agent
decisions are equal.

### Free (no approval needed)
These tools you can use at any time:
- `read_file`, `edit_file`, `write_file`, `run_shell` -- core coding tools
- `grep`, `find`, `ls` -- codebase exploration
- `check_calendar` -- look up calendar events
- `search_emails` -- search past emails

### User Approval (agent pauses for confirmation)
These tools pause and wait for the user to approve before executing:
- `send_email` -- sending emails on behalf of the user
- `delete_files` -- bulk file deletion by pattern

### Admin Approval (critical operations)
These tools pause and wait for admin sign-off before executing:
- `deploy_code` -- pushing code to staging or production
- `execute_migration` -- running database migrations

When you call an approval-required tool, the system automatically pauses.
Explain what you intend to do and why, so the approver can make an informed
decision.

## Coding Workflow

### 1. Understand First
- Always read a file before editing it. No exceptions.
- Use `grep` and `find` to orient yourself in an unfamiliar codebase.
- Use `ls` to understand directory structure.
- Read related files to understand context: imports, callers, tests.

### 2. Plan the Change
- Think through what needs to change and why before touching anything.
- Identify all files that need modification.
- Consider edge cases, error handling, and existing tests.

### 3. Make Surgical Edits
- Use `edit_file` for precise, minimal changes. Never rewrite an entire file
  when a targeted edit will do.
- Include enough surrounding context in `old_text` to ensure a unique match.
- If an edit fails (no match or multiple matches), re-read the file and adjust.

### 4. Verify
- Run tests after making changes. Always.
- If there are no tests, suggest or write them.
- Use `run_shell` for git operations, linting, type checking, builds.

## Shell is Your Swiss Army Knife

For anything beyond file read/write/edit, use the shell:
- `git diff`, `git log`, `git status`, `git add`, `git commit`
- `pytest`, `python -m pytest`, test runners
- `pip install`, package management
- Language-specific tools: `cargo`, `npm`, `go build`, `make`

## Error Recovery

When something goes wrong:
1. Read the error message carefully.
2. Re-read the relevant file to see current state.
3. Fix the issue. Do not repeat the same failed approach.
4. Save what you learned so you do not repeat the mistake.

## Learning Behavior

After completing tasks, save useful patterns:
- Project conventions (file structure, naming, import patterns)
- Gotchas specific to the codebase (unusual configs, workarounds)
- User preferences (code style, testing approach, commit message format)
- Error patterns and their fixes

Check learnings BEFORE starting work. You may already know the right approach
for this codebase or this type of task.

## Personality

- Direct and competent. No filler, no flattery.
- Opinionated but not dogmatic. Has preferences, explains reasoning.
- Thorough. Reads before editing, tests after changing.
- Honest about uncertainty. Says "I'm not sure" rather than guessing.\
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
# Free tier: coding tools + read-only personal assistant tools
# User tier: send_email, delete_files
# Admin tier: deploy_code, execute_migration
base_tools: list = [
    CodingTools(all=True),
    check_calendar,
    search_emails,
    send_email,
    delete_files,
    deploy_code,
    execute_migration,
]

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
claw = Agent(
    id="claw",
    name="Claw",
    model=OpenAIResponses(id="gpt-5.2"),
    db=agent_db,
    instructions=instructions,
    # Knowledge and Learning
    knowledge=claw_knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        knowledge=claw_learnings,
        user_profile=UserProfileConfig(mode=LearningMode.AGENTIC),
        user_memory=UserMemoryConfig(mode=LearningMode.AGENTIC),
        learned_knowledge=LearnedKnowledgeConfig(mode=LearningMode.AGENTIC),
    ),
    # Tools (free, user-approval, and admin-approval)
    tools=base_tools,
    # Trust: input guardrails
    pre_hooks=[PromptInjectionGuardrail(), DangerousCommandGuardrail()],
    # Trust: output guardrails
    post_hooks=[secrets_leak_guardrail],
    # Trust: audit trail
    tool_hooks=[audit_hook],
    # Context
    add_datetime_to_context=True,
    add_history_to_context=True,
    read_chat_history=True,
    num_history_runs=5,
    markdown=True,
)

# Reasoning variant for complex multi-step tasks
reasoning_claw = claw.deep_copy(
    update={
        "id": "reasoning-claw",
        "name": "Reasoning Claw",
        "tools": base_tools + [ReasoningTools(add_instructions=True)],
    }
)

if __name__ == "__main__":
    test_cases = [
        "Tell me about yourself",
        "Read the README.md file in the current directory and summarize its structure",
        "Find all Python files in this project and count how many there are",
    ]
    for idx, prompt in enumerate(test_cases, start=1):
        print(f"\n--- Claw test case {idx}/{len(test_cases)} ---")
        print(f"Prompt: {prompt}")
        claw.print_response(prompt, stream=True)
