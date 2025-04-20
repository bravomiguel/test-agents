from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.utils.nodes import call_llm, human_review_node, web_search_tool, human_assistance_tool
from agents.utils.state import State
from agents.utils.edges import route_after_llm

# add graph state
builder = StateGraph(State)

# add nodes
builder.add_node('call_llm', call_llm)
builder.add_node('web_search_tool', web_search_tool)
builder.add_node('human_review_node', human_review_node)
builder.add_node('human_assistance_tool', human_assistance_tool)

# add edges
builder.add_edge(START, 'call_llm')
builder.add_conditional_edges('call_llm', route_after_llm)
# builder.add_conditional_edges('call_llm', should_continue)
builder.add_edge('web_search_tool', 'call_llm')
builder.add_edge('human_assistance_tool', 'call_llm')

# add memory
memory = MemorySaver()

# compile graph
# graph = builder.compile(checkpointer=memory)
graph = builder.compile()
