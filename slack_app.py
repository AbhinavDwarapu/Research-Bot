import os
import re
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent import run_agent

load_dotenv()


app = App(token=os.environ["SLACK_BOT_TOKEN"])

def ack_mention(ack):
    ack()

def run_research(event, say, client):
    question = re.sub(r"<@[A-Z0-9]+>", "", event["text"]).strip()

    placeholder = say(text="🔎 Researching, this takes a minute…",
                      thread_ts=event["ts"])

    briefing = run_agent(question)

    client.chat_update(
        channel=placeholder["channel"],
        ts=placeholder["ts"],
        text=briefing,
    )

app.event("app_mention")(ack=ack_mention, lazy=[run_research])


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    print("⚡ Research bot is running. Mention it in Slack.")
    handler.start()