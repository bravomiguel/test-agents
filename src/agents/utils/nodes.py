from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command, interrupt
from datetime import datetime

from agents.utils.state import JokeSubjectState, OverallJokeState, State
from agents.utils.tools import human_assistance, web_search
from agents.utils.prompts import (
    EXTRACT_TOPIC_PROMPT,
    GENERATE_JOKE_PROMPT,
    GENERATE_SUBJECTS_PROMPT,
    MODEL_SYSTEM_PROMPT,
    JOKE_ROUTER_PROMPT,
    SELECT_BEST_JOKE_PROMPT,
)

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


# reject joke request node
def reject_joke_request(state: OverallJokeState):
    rejection = AIMessage(content="Sorry, I only do joke generation. Please try again.")

    return {"messages": [rejection]}


# generate joke subjects based on topic
class Subjects(BaseModel):
    subjects: list[str] = Field(
        description="List of between 2 to 5 joke subjects related to the topic."
    )


def generate_subjects(state: OverallJokeState):
    last_message = state["messages"][-1]

    extract_topic_prompt = EXTRACT_TOPIC_PROMPT.format(message=last_message.content)

    topic = llm.invoke([SystemMessage(content=extract_topic_prompt)])

    generate_subjects_prompt = GENERATE_SUBJECTS_PROMPT.format(topic=topic.content)

    response = llm.with_structured_output(Subjects).invoke(
        [SystemMessage(content=generate_subjects_prompt)]
    )

    # reset jokes to empty list
    state["jokes"] = []

    # give user feedback
    feedback = AIMessage(
        content=f"I will generate jokes about {topic.content}, and then tell you the best one."
    )

    return {"subjects": response.subjects, "messages": [feedback]}


# generate joke for each subject
class Joke(BaseModel):
    joke: str = Field(description="A joke about the subject.")


def generate_joke(state: JokeSubjectState):
    generate_joke_prompt = GENERATE_JOKE_PROMPT.format(subject=state["subject"])

    response = llm.with_structured_output(Joke).invoke(
        [SystemMessage(content=generate_joke_prompt)]
    )

    return {"jokes": [response.joke]}


# select best joke
class BestJokeId(BaseModel):
    id: int = Field(description="Index of the best joke, starting with 0")


def select_best_joke(state: OverallJokeState):
    best_joke_prompt = SELECT_BEST_JOKE_PROMPT.format(jokes=state["jokes"])

    response = llm.with_structured_output(BestJokeId).invoke(
        [SystemMessage(content=best_joke_prompt)]
    )

    return {"best_joke": state["jokes"][response.id]}


# tell joke
def tell_best_joke(state: OverallJokeState):
    best_joke = AIMessage(content=state["best_joke"])

    return {"messages": [best_joke]}
