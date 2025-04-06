"""Define the state structures for the agent."""

from __future__ import annotations
from dataclasses import dataclass
from langgraph.graph import MessagesState

@dataclass
class State(MessagesState):
    name: str
    birthday: str
