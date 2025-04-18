from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# JOKE GENERATOR SCHEMAS

class RouteOutput(BaseModel):
    route: Literal["generate_joke", "reject_joke_request"] = Field(
        description="Route to follow based on user intent. Use 'generate_joke' if the user is asking for a joke. Use 'reject_joke_request' for anything else."
    )

class Subjects(BaseModel):
    subjects: list[str] = Field(
        description="List of between 2 to 5 joke subjects related to the topic."
    )

class Joke(BaseModel):
    joke: str = Field(description="A joke about the subject.")


class BestJokeId(BaseModel):
    id: int = Field(description="Index of the best joke, starting with 0")

# TODOS MANAGER SCHEMAS

class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. E.g. user expressed interest in learning French.")

class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="List of memories about the user.")

# memory type route
class UpdateMemory(BaseModel):
    update_type: Literal["user", "todo", "instructions"] = Field(
        description="The type of memory to update. Use 'user' for user profile, 'todo' for ToDo list, and 'instructions' for instructions on how to update the ToDo list."
    )

# user profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with."""
    name: Optional[str] = Field(description="The name of the user.", default=None)
    location: Optional[str] = Field(description='Where the user lives. Include place and state name, e.g. Austin, TX', default=None)
    job: Optional[str] = Field(description="The user's job. Include company name and title if possible, e.g. Software Engineer at Google", default=None)
    connections: list[str] = Field(description="List of people user knows. Include person name and relationship if possible, e.g. John Doe, Brother", default_factory=list)
    interests: list[str] = Field(description="List of the user's interests, hobbies, and passions", default_factory=list)

# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    created_at: datetime = Field(description="When the task was created.")
    updated_at: Optional[datetime] = Field(description="When the task was last updated.", default=None)
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).")
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )

# instructions schema
class Instructions(BaseModel):
    instructions: str = Field(description="Instructions for updating the ToDo list.")

