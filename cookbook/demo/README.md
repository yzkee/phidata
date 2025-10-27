# AgentOS Demo

This cookbook show's how to run a multi-agent system using the AgentOS. It covers the following topics:

- Building Agents, Multi-Agent Teams and Agentic Workflows
- Running Agents, Teams and Workflows as an API using the AgentOS
- Connecting to the AgentOS UI

## Setup

> Note: Fork and clone the repository if needed

### 1. Create a virtual environment

```shell
uv venv .demoenv --python 3.12
source .demoenv/bin/activate
```

### 2. Install libraries

```shell
uv pip install -r cookbook/demo/requirements.txt
```

### 3. Run Postgres with PgVector

We'll use postgres for storing session, memory and knowledge. Install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) and run the following command to start a postgres container with PgVector.

```shell
./cookbook/scripts/run_pgvector.sh
```

OR use the docker run command directly:

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

### 4. Export API Keys

We recommend using claude-sonnet-4-5 for your agents, but you can use any Model you like.

```shell
export ANTHROPIC_API_KEY=***
```

### 5. Run the demo AgentOS

```shell
python cookbook/demo/run.py
```

### 6. Connect to the AgentOS UI

Open [os.agno.com](https://os.agno.com/) and connect to http://localhost:7777 to interact with the demo AgentOS.

---

## Additional Resources

For more information, read the [Agno documentation](https://docs.agno.com).
