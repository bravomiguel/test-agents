from langgraph.graph import END
from langgraph.types import Command, interrupt

from agents.utils.state import State


def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "web_search_tool"
    return END


def route_after_llm(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if (
        last_message.tool_calls
        and last_message.tool_calls[-1].get("name") == "human_assistance"
    ):
        return "human_assistance_tool"
    elif last_message.tool_calls:
        return "human_review_node"
    return END
