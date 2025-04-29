![logo](docs/pics/frontpage.png)

---

## 🚀 Overview

This package offers a framework for researchers to map and quantify interactions between students and LLM-based tutors in educational settings. It supports structured, objective evaluation through classification, simulation, and visualization tools, and is designed for flexible use across tasks of any scale. The framework accommodates both researchers analyzing pre-collected, annotated data and those starting from scratch, providing modular support through each step of the evaluation process.

The package is designed to:

- Provide a customized framework for classification, evaluation, and fine-tuning
- Simulate student–tutor interactions using role-based prompts and seed messages when real data is unavailable
- Initiate an interface with locally hosted, open-source models (e.g., via LM Studio or Hugging Face)
- Log interactions in structured formats (JSON/CSV) for downstream analysis
- Train and applu classifiers to predict customized interaction classes and visualize patterns across conversations

Overview of the system architecture:

![flowchart](docs/pics/new_flowchart.png)

---

## ⚙️ Installation

```bash
pip install educhateval
```

## 🤗 Integration 
Note that the framework and dialogue generation is integrated with [LM Studio](https://lmstudio.ai/), and the wrapper and classifiers with [Hugging Face](https://huggingface.co/).


## 📖 Documentation

| **Documentation** | **Description** |
|-------------------|-----------------|
| 📚 [User Guide](https://laurawpaaby.github.io/EduChatEval/user_guides/guide/) | Instructions on how to run the entire pipeline provided in the package |
| 💡 [Prompt Templates](https://laurawpaaby.github.io/EduChatEval/user_guides/frameworks/) | Overview of system prompts, role behaviors, and instructional strategies |
| 🧠 [API References](https://laurawpaaby.github.io/EduChatEval/api/api_frame_gen/) | Full reference for the `educhateval` API: classes, methods, and usage |
| 🤔 [About](https://laurawpaaby.github.io/EduChatEval/about/) | Learn more about the thesis project, context, and contributors |


## ⚙️ Usage
```python
from pathlib import Path
from educhateval import FrameworkGenerator, 
                        DialogueSimulator,
                        PredictLabels,
                        Visualizer
```

**1.** Generate Label Framework
```python
generator = FrameworkGenerator(
    model_name="llama-3.2-3b-instruct",
    api_url="http://localhost:1234/v1/completions"
)

df_4 = generator.generate_framework(
    prompt_path="outline_prompts/prompt_default_4types.py",
    num_samples=200
)

filtered_df = generator.filter_with_classifier(
    train_data="data/tiny_labeled_default.csv",
    synth_data=df_4
)
```

**2.** Synthesize Interaction
```python
simulator = DialogueSimulator(
    backend="mlx",
    model_id="mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
)

seed_message = "Hi, can you please help me with my English course?"

# Simulate a single student-tutor dialogue with a custom YAML file
df_single = simulator.simulate_dialogue(
    mode="general_task_solving",
    turns=10,
    seed_message_input=seed_message,
    custom_prompt_file=Path("prompts/my_custom_prompts.yaml")
)
```

**3.** Classify and Predict
```python
predictor = PredictLabels(model_name="distilbert/distilroberta-base")

annotaded_df = predictor.run_pipeline(
    train_data=filtered_df,
    new_data=df_single,
    text_column="text",
    label_column="category",
    columns_to_classify=["student_msg", "tutor_msg"],
    split_ratio=0.2
)
```

**4.** Visualize
```python
viz = Visualizer()

summary = viz.create_summary_table(
    df=annotaded_df,
    label_columns=["predicted_labels_student_msg", "predicted_labels_tutor_msg"]
)

viz.plot_category_bars(
    df=annotaded_df,
    label_columns=["predicted_labels_student_msg", "predicted_labels_tutor_msg"],
    use_percent=True,
    title="Distribution of Predicted Classes"
)

viz.plot_turn_trends(
    df=annotaded_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg",
    title="Category Distribution over Turns"
)

viz.plot_history_interaction(
    df=annotaded_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg",
    focus_agent="student",
    use_percent=True
)
```
--- 


## 🫶🏼 Acknowdledgement 

This project builds on existing tools and ideas from the open-source community. While specific references are provided within the relevant scripts throughout the repository, the key sources of inspiration are also acknowledged here to highlight the contributions that have shaped the development of this package.

- *Constraint-Based Data Generation – Outlines Package*: [Willard, Brandon T. & Louf, Rémi (2023). *Efficient Guided Generation for LLMs.*](https://arxiv.org/abs/2307.09702) 

- *Chat Interface and Wrapper – Textual*: [McGugan, W. (2024, Sep). *Anatomy of a Textual User Interface.*](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five)

- *Package Design Inspiration*: [Thea Rolskov Sloth & Astrid Sletten Rybner](https://github.com/DaDebias/genda-lens)  

- *Code Debugging and Conceptual Feedback*:
  [Mina Almasi](https://pure.au.dk/portal/da/persons/mina%40cc.au.dk) and [Ross Deans Kristensen-McLachlan](https://pure.au.dk/portal/da/persons/rdkm%40cc.au.dk)



## 📬 Contact

Made by **Laura Wulff Paaby**  
Feel free to reach out via:

- 🌐 [LinkedIn](https://www.linkedin.com/in/laura-wulff-paaby-9131a0238/)
- 📧 [laurapaaby18@gmail.com](mailto:202806616@post.au.dk)
- 🐙 [GitHub](https://github.com/laurawpaaby) 

---



## Complete overview:
``` 
├── data/                                  
│   ├── generated_dialogue_data/           # Generated dialogue samples
│   ├── generated_tuning_data/             # Generated framework data for fine-tuning 
│   ├── logged_dialogue_data/              # Logged real dialogue data
│   ├── Final_output/                      # Final classified data 
│
├── Models/                                # Folder for trained models and checkpoints (ignored)
│
├── src/educhateval/                       # Main source code for all components
│   ├── chat_ui.py                         # CLI interface for wrapping interactions
│   ├── descriptive_results/               # Scripts and tools for result analysis
│   ├── dialogue_classification/           # Tools and models for dialogue classification
│   ├── dialogue_generation/               
│   │   ├── agents/                        # Agent definitions and role behaviors
│   │   ├── models/                        # Model classes and loading mechanisms
│   │   ├── txt_llm_inputs/               # System prompts and structured inputs for LLMs
│   │   ├── chat_instructions.py          # System prompt templates and role definitions
│   │   ├── chat_model_interface.py       # Interface layer for model communication
│   │   ├── chat.py                       # Main script for orchestrating chat logic
│   │   └── simulate_dialogue.py          # Script to simulate full dialogues between agents
│   ├── framework_generation/            
│   │   ├── outline_prompts/              # Prompt templates for outlines
│   │   ├── outline_synth_LMSRIPT.py      # Synthetic outline generation pipeline
│   │   └── train_tinylabel_classifier.py # Training classifier on manually made true data
│
├── .python-version                       # Python version file for (Poetry)
├── poetry.lock                           # Locked dependency versions (Poetry)
├── pyproject.toml                        # Main project config and dependencies
``` 