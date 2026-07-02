# Research Bot

A LangGraph agent that researches a question by looping **search → write → critique** until the draft is good enough (or 3 iterations). It uses Tavily for web search and an OpenAI model to write and self-critique the briefing. Runnable from the CLI or as a Slack bot.

## Local documents (RAG)

Drop `.md` or `.txt` files into `docs/` and they are indexed into an in-memory vector store on startup (chunked, embedded with `text-embedding-3-small`). Each search pass retrieves the top 3 matching chunks and feeds them to the writer alongside the web results, cited like any other source. The index is rebuilt on every startup, so edits to `docs/` are picked up by simply restarting.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

CLI:

```bash
python agent.py "your research question"
```

Slack bot (Socket Mode — mention the bot in a channel):

```bash
python slack_app.py
```

## .env

```env
OPENAI_API_KEY=        # OpenAI API key
TAVILY_API_KEY=        # Tavily search API key
SLACK_BOT_TOKEN=       # Slack bot token (xoxb-…), only for slack_app.py
SLACK_APP_TOKEN=       # Slack app-level token (xapp-…), only for slack_app.py

# Optional — LangSmith tracing
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=
```
