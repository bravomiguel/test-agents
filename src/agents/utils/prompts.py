# model system prompt
MODEL_SYSTEM_PROMPT = """
You are a helpful assistant.

Current Time: {time}
"""

JOKE_ROUTER_PROMPT = """
You are a router that decides if a user is asking for a joke.

Instructions:
- If the message is clearly a request for a joke, route to "joke".
- For anything else (e.g., questions, facts, opinions), route to "reject".

Return a JSON object with a single key: "route".

Message: "{last_message}"
"""
