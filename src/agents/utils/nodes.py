from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command, interrupt
from datetime import datetime

from agents.utils.state import OverallJokeState, State
from agents.utils.tools import human_assistance, web_search
from agents.utils.prompts import MODEL_SYSTEM_PROMPT, JOKE_ROUTER_PROMPT

llm = ChatOpenAI(model="gpt-4o-mini")

# WEB SEARCH AGENT NODES

# llm with web search tools
llm_with_tools = llm.bind_tools([web_search, human_assistance])

sys_msg = SystemMessage(
    content=MODEL_SYSTEM_PROMPT.format(time=datetime.now().isoformat())
)


def call_llm(state: State):
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}


# human review node
class HumanReviewResponse(TypedDict):
    action: Literal["continue", "update", "feedback"]
    data: dict


def human_review_node(state: State):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]

    human_review: HumanReviewResponse = interrupt(
        {
            "question": "Is this correct?",
            "tool_call": tool_call,
        }
    )

    # Check if we're getting the simplified format
    if "action" in human_review and human_review["action"] == "continue":
        return Command(goto="web_search_tool")

    review_action = human_review["action"]
    review_data = human_review["data"]

    if review_action == "update":
        updated_message = {
            "role": "ai",
            "content": last_message.content,
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    # update provided by human
                    "args": review_data,
                }
            ],
            # overwrite last message
            "id": last_message.id,
        }

        return Command(goto="web_search_tool", update={"messages": [updated_message]})

    if review_action == "feedback":
        # tool message required after ai message with tool call
        tool_message = {
            "role": "tool",
            # human feedback
            "content": review_data,
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"],
        }

        return Command(goto="call_llm", update={"messages": [tool_message]})


# web search tool node
web_search_tool = ToolNode([web_search])

# human assistance tool node
human_assistance_tool = ToolNode([human_assistance])

# JOKE GENERATION AGENT NODES


# define joke router llm with structured output
class RouteOutput(BaseModel):
    route: Literal["generate_joke", "reject_joke_request"] = Field(
        description="Route to follow based on user intent. Use 'generate_joke' if the user is asking for a joke. Use 'reject_joke_request' for anything else."
    )


joke_router_llm = llm.with_structured_output(RouteOutput)


# joke router llm node
def decide_joke_route(state: OverallJokeState):
    last_message = state["messages"][-1]

    router_prompt = JOKE_ROUTER_PROMPT.format(last_message=last_message.content)

    result = joke_router_llm.invoke([SystemMessage(content=router_prompt)])

    return {"joke_route": result.route}


# generate joke node
def generate_joke(state: OverallJokeState):
    sys_msg = SystemMessage(content="You are an expert joke generator.")

    return {"messages": llm.invoke([sys_msg] + state["messages"])}


# reject joke request node
def reject_joke_request(state: OverallJokeState):
    rejection = AIMessage(content="Sorry, I only do joke generation. Please try again.")

    return {"messages": state["messages"] + [rejection]}
