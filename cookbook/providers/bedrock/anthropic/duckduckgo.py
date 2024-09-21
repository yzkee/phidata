from phi.agent import Agent, RunResponse
from phi.model.aws.anthropic import Claude
from phi.tools.duckduckgo import DuckDuckGo

agent = Agent(
    model=Claude(
        model ="anthropic.claude-3-5-sonnet-20240620-v1:0",
        name  = "AwsBedrockAnthropicClaude",
        max_tokens = 8192,
        temperature = 0.1,
        top_p  = 10,
        top_k = 10,
        stop_sequences = [ "/stop_sequence" ],      
        ),
    tools=[DuckDuckGo()],
    instructions=["responsd in a southern tone", "use your tools to search for ide"],
    # show_tool_calls=True,
    debug_mode=True,
    add_datetime_to_instructions=True,
)

# run: RunResponse = agent.run(
#     "you need to preform multiple searches. first list top 5 college football teams. then search for the mascot of the team with the most wins",
# )

# print(f"""
      
# Agent Response: {run.content}

# """)

# agent.print_response("you need to preform multiple searches. first list top 5 college football teams. then search for the mascot of the team with the most wins", stream=True)

agent.print_response("First search for the top 5 college football teams. Then search to see who is starting quaterback for University of Texas this weekend", stream=True)