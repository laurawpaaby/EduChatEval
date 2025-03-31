"""
Chat formatting using pydantic
"""

from typing import Literal
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """
    Chat msg formatting:
        role: sender of the content
            user = input, assistant = LLM output, system = initial system message only
        content: text written by role
    """

    role: Literal["user", "assistant", "system"]
    content: str


class ChatHistory(BaseModel):
    """
    Chat history formatting.
    """

    messages: list[ChatMessage]