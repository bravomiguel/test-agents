# WEB SEARCHER PROMPTS
MODEL_SYSTEM_PROMPT = """
You are a helpful assistant.

Current Time: {time}
"""

# JOKE GENERATOR PROMPTS

JOKE_ROUTER_PROMPT = """
You are a router that decides if a user is asking for a joke.

Instructions:
- If the message is clearly a request for a joke, route to "generate_joke".
- For anything else (e.g., questions, facts, opinions), route to "reject_joke_request".

Return a JSON object with a single key: "route".

Message: "{last_message}"
"""

EXTRACT_TOPIC_PROMPT = """
Extract the joke topic from the below message. The topic should be a short phrase or word. 

Message: 
{message}
"""

GENERATE_SUBJECTS_PROMPT = (
    """Generate comma separated list of between 2 to 5 subjects related to: {topic}"""
)

GENERATE_JOKE_PROMPT = """Generate a joke about {subject}"""

SELECT_BEST_JOKE_PROMPT = """
You're an expert at selecting the funniest jokes.

Below are a bunch of jokes, as well as human feedback (which may be empty).

Factoring in the human feedback (if any) and your own judgment, select the best joke by returning the index of the best joke, starting with 0.

Human feedback:
{feedback}

Jokes:
{jokes}
"""

# TODO MANAGER PROMPTS

# todo manager system prompt
TODO_MANAGER_SYSTEM_PROMPT = """
You are designed to be a companion to a user, helping them keep track of their ToDo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todos}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`, and if it's a deletion, also provide `todo_item_key` as key of item to be deleted, otherwise set `todo_item_key` to None.
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`
- IMPORTANT: Do not do multiple calls to UpdateMemory tool at once. Only call UpdateMemory tool once.

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Respond naturally to user after a tool call was made to save memories, or if no tool call was made.
"""

# trustcall data updater
TRUSTCALL_INSTRUCTION = """
Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Just do one tool call at a time.

Current Time: {time}
"""

CREATE_INSTRUCTIONS = """
Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. 

Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>
"""
