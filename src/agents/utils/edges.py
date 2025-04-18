from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END
from langgraph.store.base import BaseStore

from agents.utils.state import OverallJokeState, State, ToDoManagerState


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


def should_generate_joke(state: OverallJokeState):
    joke_route = state["joke_route"]
    if joke_route == "generate_joke":
        return "generate_subjects"
    else:
        return "reject_joke_request"


# joke generator edges


def continue_to_jokes(state: OverallJokeState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


def human_feedback_loop(state: OverallJokeState):
    if state["feedback"] == "yes":
        return "tell_best_joke"
    else:
        return "select_best_joke"


# todo manager edges


def memory_update_router(
    state: ToDoManagerState, config: RunnableConfig, store: BaseStore
) -> Literal["update_profile", "update_instructions", "update_todos", END]:
    """Route to the relevant memory update based on the model's update memory tool call"""

    # get last message
    last_message = state["messages"][-1]

    # if it's not a tool call, route to end (defensive check)
    if not last_message.tool_calls:
        return END

    # route to relevant update memory node, based on update type arg picked by the model in the tool call
    update_type = last_message.tool_calls[0].get("args", {}).get("update_type", "")

    if update_type == "user":
        return "update_profile"
    elif update_type == "todo":
        return "update_todos"
    elif update_type == "instructions":
        return "update_instructions"
    else:
        raise ValueError(f"Invalid update type: {update_type}")
