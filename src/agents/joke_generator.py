from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.utils.nodes import decide_joke_route, generate_joke, reject_joke_request
from agents.utils.state import OverallJokeState
from agents.utils.edges import should_generate_joke

# add graph state
builder = StateGraph(OverallJokeState)

# add nodes
builder.add_node("decide_joke_route", decide_joke_route)
builder.add_node("generate_joke", generate_joke)
builder.add_node("reject_joke_request", reject_joke_request)

# add edges
builder.add_edge(START, "decide_joke_route")
builder.add_conditional_edges(
    "decide_joke_route", should_generate_joke, ["generate_joke", "reject_joke_request"]
)
builder.add_edge("generate_joke", END)
builder.add_edge("reject_joke_request", END)

# add memory
memory = MemorySaver()

# compile graph
graph = builder.compile(checkpointer=memory)
