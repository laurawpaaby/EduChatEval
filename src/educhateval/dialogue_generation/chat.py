"""
Chat formatting using pydantic
Original code by Mina Almasi in https://github.com/INTERACT-LLM/Interact-LLM/
"""

from pydantic import BaseModel
from typing import Literal, List


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
    - messages: ordered list of ChatMessage items exchanged in a conversation
    """

    messages: List[ChatMessage]
