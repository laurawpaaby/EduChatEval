from mypythonpackage import DialogueLogger

logger = DialogueLogger(
    api_url="http://127.0.0.1:1234/v1/chat/completions",
    model_name="llama-3.2-3b-instruct",
    temperature=0.7,
    context="You are an AI tutor for a student trying to solve a task. Your instructions are: Answer clearly and consicely, Check for understanding, Always be short and concise.",
)

logger.start_dialogue(
    participant_id="test_user", output_path="data/logged_dialogue_data/"
)

# RUN LIKE THIS FROM TERMINAL :
# poetry run python run_diawrap.py
