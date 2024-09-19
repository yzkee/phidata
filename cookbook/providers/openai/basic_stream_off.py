from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat

import time

agent = Agent(
    model=OpenAIChat(model="gpt-4o"),
)

start_time = time.time()

run: RunResponse = agent.run("Share a healthy breakfast recipe")  # type: ignore

print(run.content)

