MODEL_SYSTEM_PROMPT = """
You are a helpful assistant.

Current Time: {time}
"""

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
