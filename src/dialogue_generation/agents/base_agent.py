### new version:
# base_agent.py

from dialogue_generation.chat_model_interface import ChatModelInterface
from dialogue_generation.chat import ChatMessage, ChatHistory


class BaseAgent:
    """
    Abstract agent class to manage chat interaction with an LLM model.
    Stores system prompt, maintains chat history, and handles message flow.
    """

    def __init__(self, name: str, system_prompt: str, model: ChatModelInterface):
        # Set the agent's name, e.g., "Student" or "Tutor"
        self.name = name

        # Store a reference to the model used by the agent (either ChatHF or ChatMLX)
        self.model = model

        # Initialize the conversation history with the agent’s system prompt
        # This tells the LLM what role it's supposed to play (student/tutor, etc.)
        self.chat_history = ChatHistory(
            messages=[ChatMessage(role="system", content=system_prompt)]
        )
        # The chat history is now a list with one message: the system prompt


    def append_user_message(self, message: str):
        # Add a new message from the "user" (ie student) role to the chat history
        # For the tutor, this will usually be the student’s last message
        # For the student, this will usually be the tutor’s last message
        self.chat_history.messages.append(ChatMessage(role="user", content=message))

    def append_assistant_message(self, message: str):
        # Add a new message from the "assistant" (ie tutor) role to the chat history
        # This is typically the output that the student just generated
        self.chat_history.messages.append(ChatMessage(role="assistant", content=message))
