# LLM OS

Lets build the LLM OS proposed by Andrej Karpathy [in this tweet](https://twitter.com/karpathy/status/1723140519554105733), [this tweet](https://twitter.com/karpathy/status/1707437820045062561) and [this video](https://youtu.be/zjkBMFhNj_g?t=2535).

Also checkout my [video](https://x.com/ashpreetbedi/status/1790109321939829139) on building the LLM OS for more information.

## Running the LLM OS:

> Note: Fork and clone this repository if needed

### 1. Create a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 2. Install libraries

```shell
pip install -r cookbook/llm_os/requirements.txt
```

### 3. Ollama installation

- This implementation uses Ollama Llama 3.1 8b, so you will need to [install ollama](https://ollama.com/download).

Once installed pull Llama3.1

```shell
ollama pull llama3.1
```

- To use Exa for research, export your EXA_API_KEY (get it from [here](https://dashboard.exa.ai/api-keys))

```shell
export EXA_API_KEY=xxx
```

### 4. Run PgVector

We use PgVector to provide long-term memory and knowledge to the LLM OS.
Please install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) and run PgVector using either the helper script or the `docker run` command.

- Run using a helper script

```shell
./cookbook/run_pgvector.sh
```

- OR run using the docker run command

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  phidata/pgvector:16
```

### 5. Run the LLM OS App

```shell
streamlit run cookbook/llm_os/app.py
```

- Open [localhost:8501](http://localhost:8501) to view your LLM OS.
- Add a blog post to knowledge base: https://blog.samaltman.com/gpt-4o
- Ask: What is ollama?
- Web search: Whats happening in france?
- Calculator: Whats 10!
- Enable shell tools and ask: is docker running?
- Enable the Research Assistant and ask: write a report on the ibm hashicorp acquisition
- Enable the Investment Assistant and ask: shall i invest in nvda?

### 6. Message on [discord](https://discord.gg/4MtYHHrgA8) if you have any questions

### 7. Star ⭐️ the project if you like it.

### Share with your friends: https://git.new/llm-os
