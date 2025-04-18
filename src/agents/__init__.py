import os
import getpass

from . import web_searcher, joke_generator, todos_manager

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")
_set_env('TAVILY_API_KEY')

graphs = {
    "web_searcher": web_searcher,
    "joke_generator": joke_generator,
    "todos_manager": todos_manager
}

__all__ = ["graphs"]
