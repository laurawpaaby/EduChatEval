# simulate_dialogue.py

import argparse
import random
import yaml
from pathlib import Path
from datetime import datetime

from dialogue_generation.chat import ChatMessage
from dialogue_generation.agents.student_agent import StudentAgent
from dialogue_generation.agents.tutor_agent import TutorAgent
from dialogue_generation.models.wrap_huggingface import ChatHF
from dialogue_generation.models.wrap_micr import ChatMLX


# Argument parser

def parse_args():
    parser = argparse.ArgumentParser(description="Run simulated dialogues between student and tutor agents")
    parser.add_argument("--mode", type=str, default="general_course_exploration",
                        choices=["general_course_exploration", "deep_topic_mastery", "evaluative_feedback"])
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "mlx"])
    parser.add_argument("--model_id", type=str, default="gpt2")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser.parse_args()


# YAML loading functions

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_prompts_and_seed(mode: str):
    base = Path(__file__).parents[0] / "txt_llm_inputs"
    system_prompts = load_yaml(base / "system_prompts.yaml")
    seed_messages = load_yaml(base / "student_seed_messages.yaml")

    return system_prompts["conversation_types"][mode], random.choice(seed_messages[mode])


# Core simulation logic

def simulate_conversation(model, system_prompts: dict, seed_message: str, turns: int = 5, log_dir: Path = None):
    student = StudentAgent(model=model, system_prompt=system_prompts["student"])
    tutor = TutorAgent(model=model, system_prompt=system_prompts["tutor"])

    print("\n--- Starting Dialogue Simulation ---\n")
    
    dialogue_log = []

    # Step 1: student opens with seed message
    print(f"[Student]: {seed_message}")
    student.append_user_message(seed_message)
    tutor.append_user_message(seed_message)
    dialogue_log.append(ChatMessage(role="user", content=seed_message))

    # Step 2: tutor replies
    tutor_response = tutor.act(seed_message)
    print(f"[Tutor]: {tutor_response}")
    dialogue_log.append(ChatMessage(role="assistant", content=tutor_response))
    student.append_assistant_message(tutor_response)

    # Step 3: continue loop
    for i in range(turns - 1):
        print(f"\nTurn {i + 2}:")

        student_msg = student.act()
        print(f"[Student]: {student_msg}")
        dialogue_log.append(ChatMessage(role="user", content=student_msg))
        tutor.append_user_message(student_msg)

        tutor_msg = tutor.act()
        print(f"[Tutor]: {tutor_msg}")
        dialogue_log.append(ChatMessage(role="assistant", content=tutor_msg))
        student.append_assistant_message(tutor_msg)

    # Save log as text file
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(log_dir / f"dialogue_{timestamp}.txt", "w") as f:
            for msg in dialogue_log:
                f.write(f"[{msg.role.capitalize()}] {msg.content}\n")
        print(f"\n[INFO]: Dialogue saved to {log_dir}")


# Main

def main():
    args = parse_args()

    # Setup model
    if args.backend == "hf":
        sampling_params = {"temperature": 0.9, "top_p": 0.9, "top_k": 50}
        model = ChatHF(model_id=args.model_id, sampling_params=sampling_params)
    elif args.backend == "mlx":
        sampling_params = {"temp": 0.9, "top_p": 0.9, "top_k": 40}
        model = ChatMLX(model_id=args.model_id, sampling_params=sampling_params)
    else:
        raise ValueError("Unsupported backend")

    model.load() 

    system_prompts, seed_message = load_prompts_and_seed(args.mode)

    log_path = Path(args.log_dir) if args.log_dir else None

    simulate_conversation(
        model=model,
        system_prompts=system_prompts,
        seed_message=seed_message,
        turns=args.turns,
        log_dir=log_path
    )


if __name__ == "__main__":
    main()
