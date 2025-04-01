# from typing import Literal
# from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.types import Command, interrupt

from agent.utils.state import State
from agent.utils.tools import web_search

# llm node
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([web_search])

sys_msg = SystemMessage(content="You are a helpful assistant.")

def call_llm(state: State):
    return {'messages': llm_with_tools.invoke([sys_msg] + state['messages'])}

# # human review node
# class HumanReviewResponse(TypedDict):
#     action: Literal['continue', 'update', 'feedback']
#     data: dict

# def human_review_node(state: State):
#     last_message = state['messages'][-1]
#     tool_call = last_message.tool_calls[-1]

#     human_review: HumanReviewResponse = interrupt(
#         {
#             'question': 'Is this correct?',
#             'tool_call': tool_call,
#         }
#     )

#     review_action = human_review['action']
#     review_data = human_review['data']

#     if review_action == 'continue':
#         return Command(goto='tools')

#     elif review_action == 'update':
#         updated_message = {
#             'role': 'ai',
#             'content': last_message.content,
#             'tool_calls': [
#                 {
#                     'id': tool_call['id'],
#                     'name': tool_call['name'],
#                     # update provided by human
#                     'args': review_data,
#                 }
#             ],
#             # overwrite last message
#             'id': last_message.id,
#         }

#         return Command(goto='tools', update={'messages': [updated_message]})

#     elif review_action == 'feedback':
#         # tool message required after ai message with tool call
#         tool_message = {
#             'role': 'tool',
#             # human feedback
#             'content': review_data,
#             'name': tool_call['name'],
#             'tool_call_id': tool_call['id'],
#         }

#         return Command(goto='call_llm', update={'messages': [tool_message]})

# tools node
tools = ToolNode([web_search])