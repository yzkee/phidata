from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.parallel import ParallelTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[ParallelTools()],
    instructions="No need to tell me its based on your research.",
    markdown=True,
)

agent.print_response(
    "Tell me about Agno's AgentOS?",
    stream=True,
    stream_events=True,
)
