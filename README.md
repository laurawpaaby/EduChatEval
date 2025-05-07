![logo](docs/pics/frontpage.png)

---

## 🚀 Overview

This package offers a framework for researchers to map and quantify interactions between students and LLM-based tutors in educational settings. It supports structured, objective evaluation through classification, simulation, and visualization tools, and is designed for flexible use across tasks of any scale. The framework accommodates both researchers analyzing pre-collected, annotated data and those starting from scratch, providing modular support through each step of the evaluation process.

The package is designed to:

- Provide a customized framework for classification, evaluation, and fine-tuning
- Simulate student–tutor interactions using role-based prompts and seed messages when real data is unavailable
- Initiate an interface with locally hosted, open-source models (e.g., via LM Studio or Hugging Face)
- Log interactions in structured formats (JSON/CSV) for downstream analysis
- Train and apply classifiers to predict customized interaction classes and visualize patterns across conversations

Overview of the system architecture:

![flowchart](docs/pics/new_flowchart.png)
<br>
---
## 🤗 Integration 
Note that the framework and dialogue generation is integrated with [LM Studio](https://lmstudio.ai/), and the wrapper and classifiers with [Hugging Face](https://huggingface.co/).

<br>

## ⚙️ Installation

```bash
pip install educhateval
```
<br>



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
# initiate generator 
generator = FrameworkGenerator(
    model_name="llama-3.2-3b-instruct", # the model already downloaded via LM Studio
    api_url="http://localhost:1234/v1/completions" # the address of manually activated local server 
)

# apply generator to synthesize data
df_4 = generator.generate_framework(
    prompt_path="../templates/prompt_default_4types.py", # path to prompt template, can also be a direct dictionary
    num_samples=200                                      # number of samples per category to simulate
)

# quality check and filter the data with classifier trained on a few true examples
filtered_df = generator.filter_with_classifier(
    train_data="../templates/manual_labeled.csv", # manually labeled training data
    synth_data=df_4                               # the data to quality check
)
```

**2.** Synthesize Interaction
```python
# initiate simulater
simulator = DialogueSimulator(
    backend="mlx",                                       # choose either HF or MLX driven setup
    model_id="mlx-community/Qwen2.5-7B-Instruct-1M-4bit" # load model
)

# define seed_message and prompt scheme + mode
custom_prompts = {
    "conversation_types": { 
        "general_task_solving": { # the mode
            "student": "You are a student asking for help with your Biology homework.",
            "tutor": "You are a helpful tutor assisting a student. Provide short precise answers."
        },
    }
}
prompt = custom_prompts["conversation_types"]["general_task_solving"]

seed_message = "I'm trying to understand some basic concepts of human biology, can you help?" 

# Simulate the student-tutor dialogue
df_sim = simulator.simulate_dialogue(
    mode="general_task_solving",
    turns=10,                       # number of turns 
    seed_message_input=seed_message
    system_prompts=prompt
)


```

**3.** Classify and Predict
```python
# initiate module to classify and predict labels
predictor = PredictLabels(model_name="distilbert/distilroberta-base") # model to be trained and used for predictions

annotaded_df = predictor.run_pipeline(
    train_data=filtered_df,         # the annotated data for training above
    new_data=df_sim,                # the generated dialogues 
    text_column="text",
    label_column="category",
    columns_to_classify=["student_msg", "tutor_msg"],
    split_ratio=0.2
)
```

**4.** Visualize
```python
# initiate the module for descriptive visualizations 
viz = Visualizer()

# table of predicted categories (n, %) 
summary = viz.create_summary_table(
    df=annotaded_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg"
)

# bar chart matching the table
viz.plot_category_bars(
    df=annotaded_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg"
)

# line plot of predicted categories over turns
viz.plot_turn_trends(
    df=annotaded_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg"
)

# bar chart over sequential category dependencies between agents
viz.plot_history_interaction(
    df=annotaded_df,
    student_col="predicted_labels_student_msg",
    tutor_col="predicted_labels_tutor_msg",     # only one requiring both student and tutor data
    focus_agent="student"                      # the agent to visualize category dependencies for
)
```
<br>

<br>
--- 
## 📖 Documentation

| **Documentation** | **Description** |
|-------------------|-----------------|
| 📚 [User Guide](https://laurawpaaby.github.io/EduChatEval/user_guides/guide/) | Instructions on how to run the entire pipeline provided in the package |
| 💡 [Prompt Templates](https://laurawpaaby.github.io/EduChatEval/user_guides/frameworks/) | Overview of system prompts, role behaviors, and instructional strategies |
| 🧠 [API References](https://laurawpaaby.github.io/EduChatEval/api/api_frame_gen/) | Full reference for the `educhateval` API: classes, methods, and usage |
| 🤔 [About](https://laurawpaaby.github.io/EduChatEval/about/) | Learn more about the thesis project, context, and contributors |
<br>

<br>
---


## 📬 Contact

The package is made by **Laura Wulff Paaby**  
Feel free to reach out via:

- 🌐 [LinkedIn](https://www.linkedin.com/in/laura-wulff-paaby-9131a0238/)
- 📧 [laurapaaby18@gmail.com](mailto:laurapaaby18@gmail.com)
- 🐙 [GitHub](https://github.com/laurawpaaby) 
<br>

<br>


## 🫶🏼 Acknowdledgement 

This project builds on existing tools and ideas from the open-source community. While specific references are provided within the relevant scripts throughout the repository, the key sources of inspiration are also acknowledged here to highlight the contributions that have shaped the development of this package.

- *Constraint-Based Data Generation – Outlines Package*: [Willard, Brandon T. & Louf, Rémi (2023). *Efficient Guided Generation for LLMs.*](https://arxiv.org/abs/2307.09702) 

- *Chat Interface and Wrapper – Textual*: [McGugan, W. (2024, Sep). *Anatomy of a Textual User Interface.*](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five)

- *Package Design Inspiration*: [Thea Rolskov Sloth & Astrid Sletten Rybner](https://github.com/DaDebias/genda-lens)  

- *Code Debugging and Conceptual Feedback*:
  [Mina Almasi](https://pure.au.dk/portal/da/persons/mina%40cc.au.dk) and [Ross Deans Kristensen-McLachlan](https://pure.au.dk/portal/da/persons/rdkm%40cc.au.dk)

<br>

<br>

--- 


## Complete overview:
``` 
├── data/                                  
│   ├── generated_dialogue_data/           # Generated dialogue samples
│   ├── generated_tuning_data/             # Generated framework data for fine-tuning 
│   ├── logged_dialogue_data/              # Logged real dialogue data
│   ├── Final_output/                      # Final classified data 
│   ├── templates/                         # Prompt and seed templates
│
├── models/                                # (ignored) Folder for trained models and checkpoints (ignored)
│
├── src/educhateval/                       # Main source code for all components
│   ├── chat_ui.py                         # CLI interface for wrapping interactions
│   ├── classification_utils.py            # Functions to run the different classificiation models deployed
│   ├── core.py                            # Main script behind package wrapping all functions as callable classes
│   ├── descriptive_results/               # Scripts and tools for result analysis
│   ├── dialogue_classification/           # Tools and models for dialogue classification
│   ├── dialogue_generation/               
│   │   ├── agents/                        # Agent definitions and role behaviors
│   │   ├── models/                        # Model classes and loading mechanisms
│   │   ├── txt_llm_inputs/                # Prompt loading functions
│   │   ├── chat_model_interface.py        # Interface layer for model communication
│   │   ├── chat.py                        # Script for orchestrating chat logic
│   │   └── simulate_dialogue.py           # Script to simulate full dialogues between agents
│   ├── framework_generation/            
│   │   ├── outline_prompts/               # Prompt templates for outlines
│   │   ├── outline_synth_LMSRIPT.py       # Synthetic outline generation pipeline
│   │   └── train_tinylabel_classifier.py  # Training small classifier on manually made true data
│
├── .python-version                        # Python version file for (Poetry)
├── poetry.lock                            # Locked dependency versions (Poetry)
├── pyproject.toml                         # Main project config and dependencies
``` 