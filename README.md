<div align="center" id="top">
  <a href="https://docs.agno.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg">
      <img src="https://agno-public.s3.us-east-1.amazonaws.com/assets/logo-light.svg" alt="Agno">
    </picture>
  </a>
</div>
<div align="center">
  <a href="https://docs.agno.com">📚 Documentation</a> &nbsp;|&nbsp;
  <a href="https://docs.agno.com/examples/introduction">💡 Examples</a> &nbsp;|&nbsp;
  <a href="https://www.agno.com/?utm_source=github&utm_medium=readme&utm_campaign=agno-github&utm_content=header">🏠 Website</a> &nbsp;|&nbsp;
</div>

## What is Agno?

</b>[Agno](https://docs.agno.com) is a high-performance framework and runtime for multi-agent systems. Use it to build, run and manage multi-agent systems in your cloud.</b>

---

Agno is the fastest python framework for building agents with memory, knowledge, session management, human in the loop and MCP support. You can put agents together as multi-agent teams or step-based agentic workflows.

Here’s an example of an Agent that connects to any MCP server, manages conversation history and state in a database, and is served using a FastAPI application that you can connect to the [AgentOS UI](https://os.agno.com).

```python agno_agent.py
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.os import AgentOS
from agno.tools.mcp import MCPTools

# ************* Create Agent *************
agno_agent = Agent(
    name="Agno Agent",
    model=Claude(id="claude-sonnet-4-5"),
    # Add a database to the Agent
    db=SqliteDb(db_file="agno.db"),
    # Add the Agno MCP server to the Agent
    tools=[MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")],
    # Add the previous session history to the context
    add_history_to_context=True,
    markdown=True,
)


# ************* Create AgentOS *************
agent_os = AgentOS(agents=[agno_agent])
# Get the FastAPI app for the AgentOS
app = agent_os.get_app()

# ************* Run AgentOS *************
if __name__ == "__main__":
    agent_os.serve(app="agno_agent:app", reload=True)
```

The real advantage of Agno is its [AgentOS](https://docs.agno.com/agent-os/introduction) runtime:

1. You get a pre-built FastAPI app for running your agents, teams and workflows, meaning you start building your AI product on day one. This is a remarkable advantage over other solutions.
2. You also get a UI that connects directly to the pre-built FastAPI app. Use it to test, monitor and manage your system. This gives you unmatched visibility and control.
3. Your AgentOS runs in your cloud and you get complete privacy because no data ever leaves your system. This is incredible for security conscious enterprises that can't send data to external services.

Here's how the AgentOS UI looks like:

https://github.com/user-attachments/assets/feb23db8-15cc-4e88-be7c-01a21a03ebf6

For organizations building agents, Agno provides the complete solution. You get the fastest framework for building agents (speed of development and execution), a pre-built FastAPI app that get you building product on day one, and a control plane for managing your system.

We bring a novel architecture that no other framework provides, your AgentOS runs securely in your cloud, and the control plane connects directly to it from your browser. You don't need to send data to any external services or pay retention costs, you get complete privacy and control.

## Getting started

If you're new to Agno, follow our [quickstart](https://docs.agno.com/introduction/quickstart) to build your first Agent and run it using the AgentOS.

After that, checkout the [examples gallery](https://docs.agno.com/examples/introduction) and build real-world applications with Agno.

## Documentation, Community & More Examples

- Docs: <a href="https://docs.agno.com" target="_blank" rel="noopener noreferrer">docs.agno.com</a>
- Cookbook: <a href="https://github.com/agno-agi/agno/tree/main/cookbook" target="_blank" rel="noopener noreferrer">Cookbook</a>
- Community forum: <a href="https://community.agno.com/" target="_blank" rel="noopener noreferrer">community.agno.com</a>
- Discord: <a href="https://discord.gg/4MtYHHrgA8" target="_blank" rel="noopener noreferrer">discord</a>

## Setup Your Coding Agent to Use Agno

For LLMs and AI assistants to understand and navigate Agno's documentation, we provide an [llms.txt](https://docs.agno.com/llms.txt) or [llms-full.txt](https://docs.agno.com/llms-full.txt) file.

This file is built for AI systems to efficiently parse and reference our documentation.

### IDE Integration

When building Agno agents, using Agno documentation as a source in your IDE is a great way to speed up your development. Here's how to integrate with Cursor:

1. In Cursor, go to the "Cursor Settings" menu.
2. Find the "Indexing & Docs" section.
3. Add `https://docs.agno.com/llms-full.txt` to the list of documentation URLs.
4. Save the changes.

Now, Cursor will have access to the Agno documentation. You can do the same with other IDEs like VSCode, Windsurf etc.

## Performance

At Agno, we're obsessed with performance. Why? because even simple AI workflows can spawn thousands of Agents. Scale that to a modest number of users and performance becomes a bottleneck. Agno is designed for building highly performant agentic systems:

- Agent instantiation: ~3μs on average
- Memory footprint: ~6.5Kib on average

> Tested on an Apple M4 MacBook Pro.

While an Agent's run-time is bottlenecked by inference, we must do everything possible to minimize execution time, reduce memory usage, and parallelize tool calls. These numbers may seem trivial at first, but our experience shows that they add up even at a reasonably small scale.

### Instantiation Time

Let's measure the time it takes for an Agent with 1 tool to start up. We'll run the evaluation 1000 times to get a baseline measurement.

You should run the evaluation yourself on your own machine, please, do not take these results at face value.

```shell
# Setup virtual environment
./scripts/perf_setup.sh
source .venvs/perfenv/bin/activate
# OR Install dependencies manually
# pip install openai agno langgraph langchain_openai

# Agno
python cookbook/evals/performance/instantiate_agent_with_tool.py

# LangGraph
python cookbook/evals/performance/comparison/langgraph_instantiation.py
```

> The following evaluation is run on an Apple M4 MacBook Pro. It also runs as a Github action on this repo.

LangGraph is on the right, **let's start it first and give it a head start**.

Agno is on the left, notice how it finishes before LangGraph gets 1/2 way through the runtime measurement, and hasn't even started the memory measurement. That's how fast Agno is.

https://github.com/user-attachments/assets/ba466d45-75dd-45ac-917b-0a56c5742e23

### Memory Usage

To measure memory usage, we use the `tracemalloc` library. We first calculate a baseline memory usage by running an empty function, then run the Agent 1000x times and calculate the difference. This gives a (reasonably) isolated measurement of the memory usage of the Agent.

We recommend running the evaluation yourself on your own machine, and digging into the code to see how it works. If we've made a mistake, please let us know.

### Conclusion

Agno agents are designed for performance and while we do share some benchmarks against other frameworks, we should be mindful that accuracy and reliability are more important than speed.

Given that each framework is different and we won't be able to tune their performance like we do with Agno, for future benchmarks we'll only be comparing against ourselves.

## Contributions

We welcome contributions, read our [contributing guide](https://github.com/agno-agi/agno/blob/v2.0/CONTRIBUTING.md) to get started.

## Telemetry

Agno logs which model an agent used so we can prioritize updates to the most popular providers. You can disable this by setting `AGNO_TELEMETRY=false` in your environment.

<p align="left">
  <a href="#top">⬆️ Back to Top</a>
</p>
