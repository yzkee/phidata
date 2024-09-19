from phi.agent import Agent, RunResponse
from phi.model.aws.anthropic import Claude
from phi.tools.duckduckgo import DuckDuckGo

agent = Agent(
    model=Claude(model="anthropic.claude-3-5-sonnet-20240620-v1:0"),
    tools=[DuckDuckGo()],
    instructions=["use your tools to search internet"],
    show_tool_calls=True,
    debug_mode=True,
)

run: RunResponse = agent.run(
    "you need to preform multiple searches. first list top 5 college football teams. then search for the mascot of the team with the most wins",
)

print(f"""
      
Agent Response: {run.content}

""")