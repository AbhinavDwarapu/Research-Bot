from dotenv import load_dotenv
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

def write(state: ResearchState) -> dict:
    sources_text = "\n\n".join(
        f"[{i+1}] {s.get('title', s['url'])} ({s['url']})\n{s.get('content', '')}"
        for i, s in enumerate(state["sources"])
    )
    critique = state.get("critique")
    revision_note = ""

    if critique and critique.missing:
        revision_note = f"\n\nThe previous draft was weak here — address it: {critique.missing}"

    prompt = (
        f"Write a concise, well-organized briefing answering this question:\n"
        f"{state['question']}\n\n"
        f"Use only these sources. Cite them inline like [1], [2]:\n\n"
        f"{sources_text}"
        f"{revision_note}"
    )

    response = llm.invoke(prompt)
    return {"draft": response.content}

def critique(state: ResearchState) -> dict:
    critique_llm = llm.with_structured_output(Critique)

    prompt = (
        f"You are a tough but fair editor reviewing a research briefing.\n\n"
        f"Question being answered:\n{state['question']}\n\n"
        f"Draft:\n{state['draft']}\n\n"
        f"Judge whether this is thorough, well-supported, and directly answers "
        f"the question. If it falls short, set is_sufficient to false and explain "
        f"specifically what's missing or weak so it can be improved."
    )

    result = critique_llm.invoke(prompt)

    return {
        "critique": result,
        "iterations": state["iterations"] + 1,
    }

def should_continue(state: ResearchState) -> str:
    if state["critique"].is_sufficient or state["iterations"] >= 3:
        return "done"
    return "retry"


builder = StateGraph(ResearchState)

builder.add_node("search", search)  
builder.add_node("write", write)
builder.add_node("critique", critique)

builder.add_edge(START, "search")
builder.add_edge("search", "write")
builder.add_edge("write", "critique")
builder.add_conditional_edges(
    "critique",
    should_continue,
    {"retry": "search", "done": END},
)

graph = builder.compile()

def run_agent(question: str) -> str:
    initial = {"question": question, "sources": [],
               "draft": "", "critique": None, "iterations": 0}
    result = graph.invoke(initial, config={"recursion_limit": 10})
    return result["draft"]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "state of solid-state batteries"
    print(run_agent(q))