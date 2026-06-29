import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import operator
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-mini")

# What the critique node returns
class Critique(BaseModel):
    is_sufficient: bool = Field(description="Is the draft good enough to ship?")
    score: int = Field(description="Quality score from 1 to 10")
    missing: str = Field(description="What's missing or weak, to guide the next search")


# What flows through the whole graph
class ResearchState(TypedDict):
    question: str                              # the user's question
    sources: Annotated[list, operator.add]     # accumulates
    draft: str                                 # replaced each loop
    critique: Critique                         # replaced each loop
    iterations: int                           

tavily = TavilySearch(max_results=5)

def search(state: ResearchState) -> dict:
    query = state.get("question")
    critique = state.get("critique")

    if critique and critique.missing:
        query = f"{query} {critique.missing}"[:380]

    results = tavily.invoke({"query": query})
    return {"sources": results["results"]}

