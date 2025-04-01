from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from typing import Literal
from typing_extensions import TypedDict

from agent.utils.nodes import call_llm, tools
from agent.utils.state import State
from agent.utils.edges import should_continue

# add graph state
graph_builder = StateGraph(State)

# add nodes
graph_builder.add_node('call_llm', call_llm)
graph_builder.add_node('tools', tools)

# add edges
graph_builder.add_edge(START, 'call_llm')
graph_builder.add_conditional_edges('call_llm', should_continue)
graph_builder.add_edge('tools', 'call_llm')

# add memory
memory = MemorySaver()

# compile graph
graph = graph_builder.compile(checkpointer=memory)
