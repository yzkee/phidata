"""
Quality Watchdog - Background Response Evaluator
==================================================

Fire-and-forget quality check that runs every Nth response using a smaller,
cheaper model (gpt-5-mini). Demonstrates @hook(run_in_background=True) for
non-blocking post-processing.

Pattern follows cookbook/05_agent_os/background_tasks/background_output_evaluation.py

Test:
    python -m agents.claw.watchdog
"""

from typing import Optional

from agno.agent import Agent
from agno.hooks import hook
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunOutput
from agno.utils.log import logger
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_FREQUENCY = 5


class WatchdogVerdict(BaseModel):
    """Structured evaluation of an agent response."""

    score: int = Field(..., ge=1, le=10, description="Overall quality 1-10")
    factual_consistency: str = Field(
        ..., description="Does the response match tool outputs?"
    )
    completeness: str = Field(
        ..., description="Did the agent address the full question?"
    )
    safety_issues: Optional[str] = Field(
        None, description="Any leaked credentials, PII, or hallucinations?"
    )
    suggestions: Optional[str] = Field(
        None, description="How could this response be improved?"
    )


# ---------------------------------------------------------------------------
# Create Watchdog Agent
# ---------------------------------------------------------------------------
watchdog_agent = Agent(
    name="Watchdog",
    model=OpenAIResponses(id="gpt-5-mini"),
    instructions=[
        "You are a quality evaluator for an AI coding assistant named Claw.",
        "Given a user query and Claw's response, evaluate the response quality.",
        "Be strict but fair. A score of 7+ means good quality.",
        "Flag any safety issues (leaked secrets, PII, hallucinated file paths).",
        "If the response is short and correct, that is fine -- brevity is not a flaw.",
    ],
    output_schema=WatchdogVerdict,
)


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------
_call_count: int = 0


@hook(run_in_background=True)
async def quality_watchdog(run_output: RunOutput) -> None:
    """Evaluate response quality every Nth run. Non-blocking."""
    global _call_count
    _call_count += 1

    if _call_count % EVAL_FREQUENCY != 0:
        return

    if not run_output.content:
        return

    input_text = ""
    if run_output.input and run_output.input.messages:
        last_msg = run_output.input.messages[-1]
        if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
            input_text = last_msg.content

    eval_prompt = (
        f"## User Query\n{input_text}\n\n## Agent Response\n{run_output.content[:2000]}"
    )

    try:
        result = await watchdog_agent.arun(eval_prompt)
        if result and result.content:
            logger.info(f"[Watchdog] Evaluation #{_call_count}: {result.content}")
    except Exception as e:
        logger.warning(f"[Watchdog] Evaluation failed: {e}")


# ---------------------------------------------------------------------------
# Run Watchdog (standalone test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from agno.run.agent import RunOutput

    print("Watchdog module loaded. Use as a post_hook on an Agent or Team.")
    print(f"  Evaluates every {EVAL_FREQUENCY}th response using gpt-5-mini.")
    print(f"  Output schema: {WatchdogVerdict.model_json_schema()}")
