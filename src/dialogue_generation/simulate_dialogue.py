####  -------- Purpose: -------- ####
# Simulate a structured dialogue between a student and tutor agent.
# The conversation is initialized with system prompts and a seed message.
# Each turn alternates between student and tutor using the given LLM backend.

####  -------- Inputs: -------- ####
# - model: backend model (e.g., HuggingFace or MLX wrapper)
# - system_prompts: dict containing system prompts for student and tutor roles
# - turns: number of dialogue turns
# - log_dir (optional): if given, saves full conversation log to a .txt file

####  -------- Outputs: -------- ####
# - Either returns a pandas DataFrame with columns ["turn", "student_msg", "tutor_msg"]
# - Or optionally saves the log as CSV with the same structure

import pandas as pd
from pathlib import Path
from datetime import datetime
from dialogue_generation.chat import ChatMessage
from dialogue_generation.agents.base_agent import ActiveAgent


def simulate_conversation(
    model,
    system_prompts: dict,
    turns: int = 5,
    seed_message_input: str = "Hi, I'm a student seeking assistance with my studies.",
    log_dir: Path = None,
    save_csv_path: Path = None,
) -> pd.DataFrame:
    """
    Simulates a conversation between student and tutor agents.
    Returns the dialogue as a pandas DataFrame. Optionally saves to a .csv or .txt.
    """
    student = ActiveAgent(model=model, system_prompt=system_prompts["student"])
    tutor = ActiveAgent(model=model, system_prompt=system_prompts["tutor"])

    print("\n--- Starting Dialogue Simulation ---\n")

    dialogue_log = []
    structured_rows = []
    seed_message = seed_message_input 

    # Turn 1: student opens
    print(f"[Student]: {seed_message}")
    student.append_user_message(seed_message)
    tutor.append_user_message(seed_message)
    dialogue_log.append(ChatMessage(role="user", content=seed_message))

    tutor_response = tutor.act(seed_message)
    print(f"[Tutor]: {tutor_response}")
    dialogue_log.append(ChatMessage(role="assistant", content=tutor_response))
    student.append_assistant_message(tutor_response)

    structured_rows.append(
        {"turn": 1, "student_msg": seed_message, "tutor_msg": tutor_response}
    )

    # Turn 2+ (n)
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

        structured_rows.append(
            {"turn": i + 2, "student_msg": student_msg, "tutor_msg": tutor_msg}
        )

    df = pd.DataFrame(structured_rows)

    # Save structured CSV
    if save_csv_path:
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"\n[INFO]: Structured dialogue saved to CSV at {save_csv_path}")

    # Optionally also save plain text log
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(log_dir / f"dialogue_{timestamp}.txt", "w") as f:
            for msg in dialogue_log:
                f.write(f"[{msg.role.capitalize()}] {msg.content}\n")
        print(f"[INFO]: Raw log saved to {log_dir}")

    return df
