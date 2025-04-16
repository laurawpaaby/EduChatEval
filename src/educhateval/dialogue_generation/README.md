## Dialogue simulation / interaction synthesisation (???) 
Scripts for creating synthetic interactions simulating a tutor/student conversation. Interactions are stored and used for POC on the classification and descriptive analysis. <br>

Ideally, this step isn't necessary, as a chat with an actual student is logged directly. 

### Run like dis
``` python
from mypythonpackage import DialogueSimulator

simulator = DialogueSimulator(backend="mlx", model_id="mlx-community/Qwen2.5-7B-Instruct-1M-4bit")

df = simulator.simulate_dialogue(
    mode="evaluative_feedback",
    turns=6,
    log_dir=Path("logs/raw"), # logged raw dialogue output 
    save_csv_path=Path("output/dialogue.csv") # stored clean csv output
)
```
