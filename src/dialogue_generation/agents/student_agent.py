# student_agent.py

from dialogue_generation.agents.base_agent import BaseAgent
from dialogue_generation.chat_model_interface import ChatModelInterface


class StudentAgent(BaseAgent):
    """
    Agent that simulates a curious student.

    The student acts by generating a follow-up question or continuation
    based on the previous tutor message.
    It uses the underlying base model to generate the next user message.
    """

    def __init__(self, model: ChatModelInterface, system_prompt: str):
        super().__init__(
            name="Student",
            system_prompt=system_prompt,
            model=model,
        )

    def act(self, input_message: str = "") -> str:
        """
        Generates a question to ask the tutor.
        """
        if input_message:
            self.append_assistant_message(input_message)

        response = self.model.generate(self.chat_history)
        self.append_user_message(response.content)
        return response.content
