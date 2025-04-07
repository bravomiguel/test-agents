"""Define the state structures for the agent."""

from __future__ import annotations
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from typing import Annotated, Literal
import operator

@dataclass
class State(MessagesState):
    name: str
    birthday: str

# overall joke state
@dataclass
class OverallJokeState(MessagesState):
    joke_route: Literal["generate_joke", "reject_joke_request"]
    # topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best_joke: str

@dataclass
class JokeSubjectState(TypedDict):
    subject: str