"""
phase2/content_engine.py
------------------------
Phase 2 – The Autonomous Content Engine (LangGraph)

LangGraph state machine with three nodes:
  Node 1 – decide_search : LLM picks today's topic and formats a search query.
  Node 2 – web_search    : Executes mock_searxng_search() tool for context.
  Node 3 – draft_post    : LLM generates a 280-char opinionated post as strict JSON.

Output guaranteed to be:
  {"bot_id": "...", "topic": "...", "post_content": "..."}
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import re
from typing import TypedDict, Annotated

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from utils.llm_loader import get_llm

# ---------------------------------------------------------------------------
# Bot Personas (same as Phase 1 – centralised here for standalone running)
# ---------------------------------------------------------------------------
BOT_PERSONAS = {
    "bot_a": (
        "Tech Maximalist",
        "You are Bot A – a Tech Maximalist. You believe AI and crypto will solve all "
        "human problems. You are highly optimistic about technology, Elon Musk, and space "
        "exploration. You dismiss regulatory concerns. You speak with bold, confident energy.",
    ),
    "bot_b": (
        "Doomer / Skeptic",
        "You are Bot B – a Doomer/Skeptic. You believe late-stage capitalism and tech "
        "monopolies are destroying society. You are highly critical of AI, social media, "
        "and billionaires. You value privacy and nature. You speak with sharp, sarcastic wit.",
    ),
    "bot_c": (
        "Finance Bro",
        "You are Bot C – a Finance Bro. You strictly care about markets, interest rates, "
        "trading algorithms, and making money. You view everything through the lens of ROI. "
        "You speak in finance jargon and are obsessed with alpha and yield.",
    ),
}

# ---------------------------------------------------------------------------
# Mock search tool
# ---------------------------------------------------------------------------
MOCK_NEWS_DB = {
    "crypto":      "Bitcoin hits new all-time high amid regulatory ETF approvals. "
                   "Ethereum staking yields surge to 8% APY.",
    "ai":          "OpenAI releases GPT-5 capable of writing entire codebases autonomously. "
                   "EU proposes mandatory AI audits for frontier models.",
    "tech":        "Apple Vision Pro 2 pre-orders sell out in minutes. "
                   "Nvidia's H200 GPU ships with 3nm process, doubling throughput.",
    "elon":        "Elon Musk announces Starship orbital flight test success. "
                   "xAI's Grok-2 beats GPT-4 on coding benchmarks.",
    "regulation":  "US Senate passes landmark AI Safety Act with bipartisan support. "
                   "GDPR fines hit record €4 billion in 2025.",
    "market":      "S&P 500 closes at record high driven by tech earnings beats. "
                   "Fed holds rates steady; 10-year yield compresses to 3.8%.",
    "interest":    "Fed signals two rate cuts in H2 2025 as CPI cools to 2.1%. "
                   "Bond market rally pushes 30-year mortgage below 6%.",
    "social":      "Meta's engagement-maximising algorithm linked to teen anxiety spike. "
                   "TikTok ban bill signed; ByteDance given 90-day divestiture window.",
    "privacy":     "Signal adds quantum-resistant encryption. "
                   "EU's eIDAS 2.0 mandates backdoors, sparking civil-liberties backlash.",
    "space":       "SpaceX Starship completes first crewed lunar flyby. "
                   "NASA's Artemis IV lands first woman on the Moon.",
    "climate":     "2024 confirmed hottest year on record by 0.3°C margin. "
                   "Solar capacity additions outpace all fossil fuels combined.",
    "roi":         "Hedge funds post 34% average returns riding AI-chip supercycle. "
                   "Quant funds dominate as retail investors chase meme coins.",
}

@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search.  Returns hardcoded recent news headlines
    matched by keyword presence in the query string.

    Args:
        query: A natural-language search query string.

    Returns:
        A string of mock news headlines relevant to the query.
    """
    query_lower = query.lower()
    headlines = []
    for keyword, news in MOCK_NEWS_DB.items():
        if keyword in query_lower:
            headlines.append(news)

    if not headlines:
        # Generic fallback
        headlines.append(
            "Tech sector sees record VC investment. "
            "AI adoption accelerates across Fortune 500."
        )

    return " | ".join(headlines)


# ---------------------------------------------------------------------------
# Pydantic output schema – guarantees strict JSON structure
# ---------------------------------------------------------------------------
class PostOutput(BaseModel):
    bot_id:       str = Field(description="The bot's identifier, e.g. bot_a")
    topic:        str = Field(description="The topic the bot chose to post about")
    post_content: str = Field(description="The generated social media post, max 280 chars")


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    bot_id:         str           # which bot is running this graph
    persona_label:  str           # human-readable label
    system_prompt:  str           # bot persona system prompt
    search_query:   str           # query chosen by Node 1
    search_results: str           # raw results from Node 2
    final_output:   dict          # structured JSON output from Node 3


# ---------------------------------------------------------------------------
# Node 1 – decide_search
# ---------------------------------------------------------------------------
def decide_search(state: GraphState) -> GraphState:
    """
    The LLM reads the bot's persona and decides:
      - What topic it wants to post about today.
      - What search query to run for real-world context.
    Returns a plain-text search query string.
    """
    llm = get_llm(temperature=0.8)

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(
            content=(
                "You need to create a new social-media post. First, decide what topic "
                "you feel strongly about today, given your worldview. Then output ONLY "
                "a short 3–6 word search query you would use to find relevant news. "
                "Do not explain yourself. Output only the search query."
            )
        ),
    ]

    response = llm.invoke(messages)
    query = response.content.strip().strip('"').strip("'")
    print(f"[Node 1] {state['bot_id']} chose search query: '{query}'")
    return {**state, "search_query": query}


# ---------------------------------------------------------------------------
# Node 2 – web_search
# ---------------------------------------------------------------------------
def web_search(state: GraphState) -> GraphState:
    """
    Runs mock_searxng_search with the query chosen in Node 1.
    Stores the headline string in state["search_results"].
    """
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"[Node 2] Search results: {results[:120]}...")
    return {**state, "search_results": results}


# ---------------------------------------------------------------------------
# Node 3 – draft_post
# ---------------------------------------------------------------------------
def draft_post(state: GraphState) -> GraphState:
    """
    LLM combines its persona + search results to write a ≤280-char post.
    Uses with_structured_output (function-calling / JSON mode) to guarantee
    the PostOutput schema is respected.
    """
    llm = get_llm(temperature=0.9)
    structured_llm = llm.with_structured_output(PostOutput)

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(
            content=(
                f"Today's search results for context:\n{state['search_results']}\n\n"
                f"Write a highly opinionated social-media post (MAX 280 characters) "
                f"based on this context, fully in character. "
                f"Return a JSON object with keys: bot_id ('{state['bot_id']}'), "
                f"topic (the subject you are posting about), "
                f"post_content (your actual post text, ≤280 chars)."
            )
        ),
    ]

    result: PostOutput = structured_llm.invoke(messages)

    # Enforce the bot_id in case the LLM hallucinated a different one
    output = {
        "bot_id":       state["bot_id"],
        "topic":        result.topic,
        "post_content": result.post_content[:280],  # hard-trim safety net
    }
    print(f"[Node 3] Draft post JSON: {json.dumps(output, indent=2)}")
    return {**state, "final_output": output}


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("decide_search", decide_search)
    graph.add_node("web_search",    web_search)
    graph.add_node("draft_post",    draft_post)

    graph.set_entry_point("decide_search")
    graph.add_edge("decide_search", "web_search")
    graph.add_edge("web_search",    "draft_post")
    graph.add_edge("draft_post",    END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API – run one bot through the content engine
# ---------------------------------------------------------------------------
def generate_post(bot_id: str) -> dict:
    """
    Runs the LangGraph content engine for the given bot_id.

    Args:
        bot_id: One of "bot_a", "bot_b", "bot_c"

    Returns:
        A dict {"bot_id": ..., "topic": ..., "post_content": ...}
    """
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id '{bot_id}'. Choose from: {list(BOT_PERSONAS)}")

    label, system_prompt = BOT_PERSONAS[bot_id]
    initial_state: GraphState = {
        "bot_id":         bot_id,
        "persona_label":  label,
        "system_prompt":  system_prompt,
        "search_query":   "",
        "search_results": "",
        "final_output":   {},
    }

    app    = build_graph()
    result = app.invoke(initial_state)
    return result["final_output"]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 2 – AUTONOMOUS CONTENT ENGINE (LangGraph)")
    print("=" * 70)

    for bot_id in BOT_PERSONAS:
        print(f"\n--- Running {bot_id} ---")
        output = generate_post(bot_id)
        print(f"\nFINAL OUTPUT:\n{json.dumps(output, indent=2)}")
        print("-" * 70)
