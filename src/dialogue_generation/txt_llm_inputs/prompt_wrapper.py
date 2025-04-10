"""
Initial inspiration:https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five
and Mina Almasi's application in: https://github.com/INTERACT-LLM/Interact-LLM/blob/main/src/interact_llm
"""

from pathlib import Path

import toml
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """
    Model for prompt data
    """

    id: str
    content: str


class SystemPrompt(Prompt):
    """
    Model for system prompt data
    """

    role: str = Field(default="system", frozen=True)


def load_prompt_by_id(
    toml_path: Path, prompt_id: str, system_prompt: bool = True
) -> Prompt | SystemPrompt | None:
    """
    Load a prompt by its ID from a TOML file and return it as either a SystemPrompt or a regular Prompt.

    Args:
        toml_path (Path): Path to the TOML file.
        prompt_id (str): The ID of the prompt to retrieve.
        system_prompt (bool): Whether to return a SystemPrompt or a regular Prompt.

    Returns:
        Prompt | SystemPrompt | None: The requested prompt if found, otherwise None.
    """
    data = toml.load(toml_path)

    for p in data.get("prompts", []):
        if p["id"] == prompt_id:
            # make data into prompt
            prompt = Prompt.model_validate(p)
            return (
                SystemPrompt(id=prompt.id, content=prompt.content)
                if system_prompt
                else prompt
            )

    # print warning and return none if no prompt with ID found
    print(
        f"[WARNING:] No prompt found with ID {prompt_id}, running without custom prompt ..."
    )
    return None