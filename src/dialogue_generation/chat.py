# chat.py

from pydantic import BaseModel
from typing import Literal, List


class ChatMessage(BaseModel):
    """
    Represents a single message in a chat conversation.
    Roles must be 'user', 'assistant', or 'system'.
    """

    role: Literal["user", "assistant", "system"]
    content: str


class ChatHistory(BaseModel):
    """
    Container for storing the full sequence of chat messages.
    Can be passed to model.generate() as the conversation context.
    """

    messages: List[ChatMessage]
