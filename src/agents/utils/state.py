"""Define the state structures for the agent."""

from __future__ import annotations
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from typing import Annotated, Literal
import operator

# WEB SEARCHER STATE
@dataclass
class State(MessagesState):
    name: str
    birthday: str

# JOKE GENERATOR STATE

# Define a custom reducer that supports both adding and resetting
def list_with_reset_reducer(current_value, new_value):
    if new_value == "__RESET__":  # Special marker to indicate reset
        return []
    elif isinstance(new_value, list):
        return current_value + new_value
    else:
        return current_value + [new_value]


# overall joke state
@dataclass
class OverallJokeState(MessagesState):
    joke_route: Literal["generate_joke", "reject_joke_request"]
    subjects: list[str]
    jokes: Annotated[list[str], list_with_reset_reducer]
    best_joke: str
    feedback: str | None = None


@dataclass
class JokeSubjectState(TypedDict):
    subject: str

# TODO MANAGER STATE
@dataclass
class ToDoManagerState(MessagesState):
    """State for the todo manager."""

