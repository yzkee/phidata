from phi.agent import Agent
from phi.model.aws.anthropic import Claude

agent = Agent(
    model=Claude(model="anthropic.claude-3-5-sonnet-20240620-v1:0"),
    description="You help people with their health and fitness goals.",
)

agent.print_response("Share a healthy breakfast recipe", stream=True)