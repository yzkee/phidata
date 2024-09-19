from phi.agent import Agent, RunResponse
from phi.model.aws.anthropic import Claude

agent = Agent(
    model=Claude(model="anthropic.claude-3-5-sonnet-20240620-v1:0"),
    description="You help people with their health and fitness goals.",
)

run: RunResponse = agent.run("Share a quick healthy breakfast recipe.")

print(run.content)