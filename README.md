![logo](docs/pics/frontpage.png)

---

## 🚀 Overview

This package offers a framework for researchers to log and classify interactions between students and LLM-based tutors in educational settings. It supports structured, objective evaluation through classification, simulation, and visualization utilities, and is designed for flexible use across tasks of any scale. The framework supports both researchers with pre-collected datasets and those operating in data-sparse contexts. It designed as a modular tool that can be integrated at any stage of the evaluation process.

**The package is designed to:**

- Synthesize a labeled classification framework using user-defined categories 
- Simulate multi-turn student–tutor dialogues via role-based prompting and structured seed messages
- Wrap direct student-tutors interaction with locally hosted LLMs through a terminal-based interface 
- Fine-tune and apply classification models to label conversational turns
- Visualize dialogue patterns with summary tables, frequency plots, temporal trends, and sequential dependencies

Overview of the system architecture:

<br>

![flowchart](docs/pics/new_flowchart.png)
<br>
---
## 🤗 Integration 
Note that the framework and dialogue generation is integrated with [LM Studio](https://lmstudio.ai/), and the wrapper and classifiers with [Hugging Face](https://huggingface.co/).

The package currently requires [`Python 3.12`](https://www.python.org/downloads/release/python-3120/) due to version constraints in core dependencies, particularly [`outlines`](https://github.com/dottxt-ai/outlines?tab=readme-ov-file#type-constraint).

<br>

## ⚙️ Installation
EduChatEval can be installed via pip from PyPI:

```bash
pip install educhateval
```

Or from [Github](https://github.com/laurawpaaby/EduChatEval/tree/main):

```bash
pip install git+https://github.com/laurawpaaby/EduChatEval.git
```

<br>


## ⚙️ Usage

Below is the simplest example of how to use the package. For more detailed and explained application examples, see the [user guides in the documentation](https://laurawpaaby.github.io/EduChatEval/user_guides/userguide_intro/) or explore the [tutorial notebooks](https://github.com/laurawpaaby/EduChatEval/tree/main/tutorials).

Import of each module:
```python
# import modules
from educhateval import FrameworkGenerator, 
                        DialogueSimulator,
                        PredictLabels,
                        Visualizer
```

**1.** Generate Label Framework
An annotated dataset of is created using downloaded LLM, LM Studio, and a prompt template of the desired labels. (1.1) 
The data is quality assessed and filtered in a few shot approach (1.2)

```python
# 1.1
# initiate generator 
generator = FrameworkGenerator(
    model_name="llama-3.2-3b-instruct", # the model already downloaded and loaded via LM Studio
    api_url="http://localhost:1234/v1/completions" # the address of locally hosted LM Studio API endpoint that handles generation requests. Consist of server host, port, and path.
)

# apply generator to synthesize data
df_4 = generator.generate_framework(
    prompt_path="../templates/prompt_default_4types.py", # path to prompt template, can also be a direct dictionary
    num_samples=200                                      # number of samples per category to simulate
)

# 1.2 
# quality check and filter the data with classifier trained on a few true examples
filtered_df = generator.filter_with_classifier(
    train_data="../templates/manual_labeled.csv", # manually labeled training data
    synth_data=df_4                               # the data to quality check
)
```

**2.** Synthesize Interaction
Dialogues between two agents, a student and a tutor, are simulated to mimic student-chatbot interactions in real deployments.
Seed message and prompts are defined to guide the agent behavior.

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
The annotaded data generated in Step 1 is used to train a classification model, which is then directly deployed to classify the messages of the dialogues from Step 2. 

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
The predicted dialogue classes of Step 3 are summarised and visualized for interpretation. 

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
- 📧 [Mail](mailto:laurapaaby18@gmail.com)
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
├── docs/                                  # Markdowns to publish with MKDocs
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
├── tutorials/                             # Tutorials on how to use the package in different settings
│
├── mkdocs.yml                             # MKDocs configuration file
├── LICENSE                                # MIT License
├── .python-version                        # Python version file for (Poetry)
├── poetry.lock                            # Locked dependency versions (Poetry)
├── pyproject.toml                         # Main project config and dependencies
│
├── models/                                # (ignored) Folder for trained models 
├── results/                               # (ignored) Folder for training checkpoints
├── site/                                  # (ignored) MKDocs files for documentation
``` 