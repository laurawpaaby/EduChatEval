# base_model_interface.py

from abc import ABC, abstractmethod
from typing import List
from dialogue_generation.chat_instructions import ChatMessage


class ChatModelInterface(ABC):
    """
    Abstract base class that defines the required interface for any chat model
    used within the chatbot interaction system.

    Implementing classes must provide:
    - A `load()` method to initialize the model and tokenizer resources.
    - A `generate()` method that accepts a list of ChatMessage objects as
      conversational context and returns a ChatMessage containing the assistant's response.

    This interface ensures that agents can interact with any underlying model
    implementation (e.g., Hugging Face, MLX) in a consistent way.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def generate(
        self, chat: List[ChatMessage], max_new_tokens: int = 3000
    ) -> ChatMessage:
        """Generate a response based on chat history."""
        pass
