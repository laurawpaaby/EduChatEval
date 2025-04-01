####  -------- Purpose: -------- ####
# Load YAML configuration files with system prompts (and optionally seed messages)
# Used to initialize simulated dialogues with correct agent instructions

####  -------- Inputs: -------- ####
# - mode: string representing the conversation type (e.g. 'deep_topic_mastery')
# - base_path (optional): custom path to where the YAML files live (defaults to local 'txt_llm_inputs')

####  -------- Outputs: -------- ####
# - Dictionary with system prompts for student and tutor

from pathlib import Path
import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_prompts_and_seed(mode: str, base_path: Path = None) -> dict:
    """
    Load system prompts for a given mode from a YAML file.
    """
    base = base_path or Path(__file__).parents[0]
    system_prompt_path = base / "system_prompts.yaml"

    system_prompts = load_yaml(system_prompt_path)
    return system_prompts["conversation_types"][mode]
