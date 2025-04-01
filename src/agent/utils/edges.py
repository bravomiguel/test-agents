from langgraph.graph import END
from langgraph.types import Command, interrupt

from agent.utils.state import State

def should_continue(state: State):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'tools'
    return END

def route_after_llm(state: State):
  if len(state['messages'].tool_calls) == 0:
    return END
  return 'human_review_node'