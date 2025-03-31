## Dialogue simulation / interaction synthesisation (???) 
Scripts for creating synthetic interactions simulating a tutor/student conversation. Interactions are stored and used for POC on the classification and descriptive analysis. <br>


Ideally, this step isn't necessary, as a chat with an actual student is logged directly. 

### Run like dis
poetry run python -m dialogue_generation.simulate_dialogue \
  --mode deep_topic_mastery \
  --backend mlx \
  --model_id mlx-community/Qwen2.5-7B-Instruct-1M-4bit \
  --turns 5 \
  --log_dir logs/deep_topic_mastery
