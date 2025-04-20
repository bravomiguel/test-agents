from typing import Literal
from uuid import uuid4
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langgraph.types import Command, interrupt
from datetime import datetime
from agents.utils.classes import Spy
from trustcall import create_extractor
import uuid

from agents.utils.state import (
    JokeSubjectState,
    OverallJokeState,
    State,
    ToDoManagerState,
)
from agents.utils.tools import (
    UpdateMemory,
    extract_tool_info,
    human_assistance,
    web_search,
)
from agents.utils.prompts import (
    CREATE_INSTRUCTIONS,
    EXTRACT_TOPIC_PROMPT,
    GENERATE_JOKE_PROMPT,
    GENERATE_SUBJECTS_PROMPT,
    MODEL_SYSTEM_PROMPT,
    JOKE_ROUTER_PROMPT,
    SELECT_BEST_JOKE_PROMPT,
    TODO_MANAGER_SYSTEM_PROMPT,
    TRUSTCALL_INSTRUCTION,
)
from agents.utils.schemas import BestJokeId, Joke, Profile, RouteOutput, Subjects, ToDo

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


# joke router llm node
def decide_joke_route(state: OverallJokeState):
    last_message = state["messages"][-1]

    router_prompt = JOKE_ROUTER_PROMPT.format(last_message=last_message.content)

    result = llm.with_structured_output(RouteOutput).invoke(
        [SystemMessage(content=router_prompt)]
    )

    return {"joke_route": result.route}


# reject joke request node
def reject_joke_request(state: OverallJokeState):
    rejection = AIMessage(content="Sorry, I only do joke generation. Please try again.")

    return {"messages": [rejection]}


# generate joke subjects based on topic
def generate_subjects(state: OverallJokeState):
    last_message = state["messages"][-1]

    extract_topic_prompt = EXTRACT_TOPIC_PROMPT.format(message=last_message.content)

    topic = llm.invoke([SystemMessage(content=extract_topic_prompt)])

    generate_subjects_prompt = GENERATE_SUBJECTS_PROMPT.format(topic=topic.content)

    response = llm.with_structured_output(Subjects).invoke(
        [SystemMessage(content=generate_subjects_prompt)]
    )

    # give user feedback
    feedback = AIMessage(
        content=f"I will generate jokes about {topic.content}, and then tell you the best one."
    )

    return {
        "subjects": response.subjects,
        "messages": [feedback],
        "feedback": None,
        "jokes": "__RESET__",
    }


# generate joke for each subject
def generate_joke(state: JokeSubjectState):
    generate_joke_prompt = GENERATE_JOKE_PROMPT.format(subject=state["subject"])

    response = llm.with_structured_output(Joke).invoke(
        [SystemMessage(content=generate_joke_prompt)]
    )

    return {"jokes": [response.joke]}


# select best joke
def select_best_joke(state: OverallJokeState):
    feedback = state.get("feedback", "")
    jokes = state.get("jokes", [])

    best_joke_prompt = SELECT_BEST_JOKE_PROMPT.format(feedback=feedback, jokes=jokes)

    response = llm.with_structured_output(BestJokeId).invoke(
        [SystemMessage(content=best_joke_prompt)]
    )

    return {"best_joke": state["jokes"][response.id]}


# human confirm node
def human_feedback(state: OverallJokeState):
    human_interrupt = interrupt(
        {
            "question": "Is this the funniest joke?",
            "best_joke": state["best_joke"],
            "jokes": state["jokes"],
        }
    )

    feedback = human_interrupt.get("feedback", "")

    if feedback.lower().startswith("y"):
        return {"feedback": "yes"}
    else:
        return {"feedback": feedback}


# tell joke
def tell_best_joke(state: OverallJokeState):
    best_joke = AIMessage(content=state["best_joke"])

    return {"messages": [best_joke]}


# TODO MANAGER NODES


# todo manager
def todo_manager(state: ToDoManagerState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response. Either updating memories or responding to the user and ending."""

    # get user id from config
    user_id = config["configurable"]["user_id"]

    # get user profile from store
    profile_object = store.search(("profile", user_id))
    if profile_object:
        user_profile = profile_object[0].value
    else:
        user_profile = None

    # get todo list from store
    todos_object = store.search(("todo", user_id))
    todos = "\n".join(
        f"{{'key': {todo.key}, 'value': {todo.value}}}" for todo in todos_object
    )

    # get instructions from store
    instructions_object = store.search(("instructions", user_id))
    if instructions_object:
        instructions = instructions_object[0].value
    else:
        instructions = None

    # add info to system prompt
    system_prompt = TODO_MANAGER_SYSTEM_PROMPT.format(
        user_profile=user_profile, todos=todos, instructions=instructions
    )

    # call llm (with update memory tool) passing in system prompt + chat history
    response = llm.bind_tools([UpdateMemory]).invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {"messages": [response]}


# update user profile node
def update_profile(state: ToDoManagerState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update user profile."""

    # get user id from config
    user_id = config["configurable"]["user_id"]

    # get user profile from store
    profile_object = store.search(("profile", user_id))

    # format user_profile for trust call extractor
    profile_object = (
        [(item.key, "Profile", item.value) for item in profile_object]
        if profile_object
        else None
    )

    # prep system prompt and chat history as input (minus last chat message which is a tool call)
    system_prompt = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=system_prompt)] + state["messages"][:-1]
        )
    )

    # create trustcall extractor for updating user profile
    profile_extractor = create_extractor(llm, tools=[Profile], tool_choice="Profile")

    # invoke trustcall extractor to update user profile based on chat history
    result = profile_extractor.invoke(
        {"messages": messages, "existing": profile_object}
    )

    # update user profile in store (item by item in profile json object)
    for response, response_metadata in zip(
        result["responses"], result["response_metadata"]
    ):
        store.put(
            ("profile", user_id),
            response_metadata.get("json_doc_id", str(uuid.uuid4())),
            response.model_dump(mode="json"),
        )

    # Update the tool call made in todo_manager, with profile updated message
    return {
        "messages": ToolMessage(
            content="Profile updated",
            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
        )
    }

    # return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}


# update todo list node
def update_todos(state: ToDoManagerState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update todo list."""

    # get user id from config
    user_id = config["configurable"]["user_id"]

    # if todo item key is provided in last message tool call args, delete item and return "item deleted"
    tool_call = state["messages"][-1].tool_calls[0]
    todo_item_key = tool_call["args"].get("todo_item_key", None)
    if todo_item_key:
        store.delete(("todo", user_id), todo_item_key)
        return {
            "messages": [
                ToolMessage(
                    content=f"Item {todo_item_key} deleted",
                    tool_call_id=tool_call["id"],
                )
            ]
        }

    # get todos from store
    todos = store.search(("todo", user_id))

    # format todos for trust call extractor
    todos = [(item.key, "ToDo", item.value) for item in todos] if todos else None

    # prep system prompt and chat history as input (minus last chat message which is a tool call)
    system_prompt = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=system_prompt)] + state["messages"][:-1]
        )
    )

    # instantiate spy to inspect tool calls made by trustcall
    spy = Spy()

    # create trustcall extractor for updating todos
    todo_extractor = create_extractor(
        llm, tools=[ToDo], tool_choice="ToDo", enable_inserts=True
    ).with_listeners(on_end=spy)

    # invoke the extractor
    result = todo_extractor.invoke({"messages": messages, "existing": todos})

    # update todos in store (item by item in todo json object)
    for response, response_metadata in zip(
        result["responses"], result["response_metadata"]
    ):
        store.put(
            ("todo", user_id),
            response_metadata.get("json_doc_id", str(uuid.uuid4())),
            response.model_dump(mode="json"),
        )

    # extract changes made by trustcall to todo list
    todos_changes = extract_tool_info(spy.called_tools, schema_name="ToDo")

    # update the tool call made by todo_manager, with todos changes message
    return {
        "messages": ToolMessage(
            content=todos_changes,
            tool_call_id=tool_call["id"],
        )
    }


# update instructions node
def update_instructions(
    state: ToDoManagerState, config: RunnableConfig, store: BaseStore
):
    """Reflect on the chat history and update instructions."""

    # get user id from config
    user_id = config["configurable"]["user_id"]

    # get existing instructions from store
    instructions = store.get(("instructions", user_id), "user_instructions")

    # prep system prompt
    system_prompt = CREATE_INSTRUCTIONS.format(
        current_instructions=instructions.value if instructions else None
    )

    # call model with system prompt and chat history to get new instructions set
    response = llm.invoke(
        [SystemMessage(content=system_prompt)]
        + state["messages"][:-1]
        + [
            HumanMessage(
                content="Please update the instructions based on the conversation"
            )
        ]
    )

    # update instructions in store
    store.put(
        ("instructions", user_id),
        "user_instructions",
        {"instructions": response.content},
    )

    # update tool call made by todo_manager, with instructions updated message
    return {
        "messages": [
            ToolMessage(
                content="Instructions updated",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }
