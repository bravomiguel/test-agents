from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from typing import Literal
from typing_extensions import TypedDict

from agent.utils.nodes import call_llm, tools, human_review_node
from agent.utils.state import State
from agent.utils.edges import should_continue, route_after_llm

# add graph state
builder = StateGraph(State)

# add nodes
builder.add_node(call_llm)
builder.add_node(tools)
builder.add_node(human_review_node)

# add edges
builder.add_edge(START, 'call_llm')
builder.add_conditional_edges('call_llm', route_after_llm)
# builder.add_conditional_edges('call_llm', should_continue)
builder.add_edge('tools', 'call_llm')


# add memory
memory = MemorySaver()

# compile graph
graph = builder.compile(checkpointer=memory)
