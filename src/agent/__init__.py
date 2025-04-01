"""New LangGraph Agent.

This module defines a custom graph.
"""

import os
import getpass

from agent.graph import graph

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")
_set_env('TAVILY_API_KEY')

__all__ = ["graph"]
