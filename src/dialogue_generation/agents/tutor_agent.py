# tutor_agent.py

from dialogue_generation.agents.base_agent import BaseAgent
from dialogue_generation.chat_model_interface import ChatModelInterface


class TutorAgent(BaseAgent):
    """
    Agent that simulates a knowledgeable and helpful tutor.

    This agent generates informative and supportive responses to questions from the student.
    """

    def __init__(self, model: ChatModelInterface, system_prompt: str):
        super().__init__(
            name="Tutor",
            system_prompt=system_prompt,
            model=model,
        )

    def act(self, input_message: str = "") -> str:
        """
        Generates a response to the student's latest question.
        """
        if input_message:
            self.append_user_message(input_message)

        response = self.model.generate(self.chat_history)
        self.append_assistant_message(response.content)
        return response.content
    

 