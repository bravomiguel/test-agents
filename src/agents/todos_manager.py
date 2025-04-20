from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


from agents.utils.nodes import (
    todo_manager,
    update_instructions,
    update_profile,
    update_todos,
)
from agents.utils.state import ToDoManagerState
from agents.utils.edges import memory_update_router

# add graph state
builder = StateGraph(ToDoManagerState)

# add nodes
builder.add_node(todo_manager)
builder.add_node(update_profile)
builder.add_node(update_instructions)
builder.add_node(update_todos)

# add edges
builder.add_edge(START, "todo_manager")
builder.add_conditional_edges(
    "todo_manager",
    memory_update_router,
    ["update_profile", "update_instructions", "update_todos", END],
)
builder.add_edge("update_profile", "todo_manager")
builder.add_edge("update_instructions", "todo_manager")
builder.add_edge("update_todos", "todo_manager")

# store for across thread memory (long term memory)
across_thread_memory = InMemoryStore()

# checkpointer for within thread memory (short term memory)
within_thread_memory = MemorySaver()

# compile graph
# graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)
graph = builder.compile()
