![logo](docs/pics/frontpage.png)


_A structured pipeline and Python package for evaluating interactive LLM tutor systems in educational settings._

---

## 🚀 Overview

This package provides an evaluation framework for analyzing interactions between students and LLM-based tutors through classification, simulation, and visualization tools.

The package is designed to:

- Provide a customized framework for classification, evaluation, and fine-tuning
- Simulate student–tutor interactions using role-based prompts and seed messages when real data is unavailable
- Initiate an interface with locally hosted, open-source models (e.g., via LM Studio or Hugging Face)
- Log interactions in structured formats (JSON/CSV) for downstream analysis
- Train and applu classifiers to predict customized interaction classes and visualize patterns across conversations

Overview of the system architecture:

![flowchart](new_flowchart.png)

---

## ⚙️ Installation

```bash
pip install educhateval
```

## ⚙️ Usage
```python
from educhateval import FrameworkGenerator, 
                        DialogueSimulator,
                        PredictLabels,
                        Visualizer

```


## 🤗 Integration 
Note that the framework and dialogue generation is integrated with [LM Studio](https://lmstudio.ai/), and the wrapper and classifiers with [Hugging Face](https://huggingface.co/)



## 📖 Documentation

| **Documentation** | **Description** |
|-------------------|-----------------|
| 📚 [User Guide](https://laurawpaaby.github.io/EduChatEval/user_guides/guide/) | Instructions on how to run the entire pipeline provided in the package |
| 💡 [Prompt Templates](https://laurawpaaby.github.io/EduChatEval/user_guides/frameworks/) | Overview of system prompts, role behaviors, and instructional strategies |
| 🧠 [API References](https://laurawpaaby.github.io/EduChatEval/api/api_frame_gen/) | Full reference for the `educhateval` API: classes, methods, and usage |
| 🤔 [About](https://laurawpaaby.github.io/EduChatEval/about/) | Learn more about the thesis project, context, and contributors |


## 🫶🏼 Acknowdledgement 
TBA


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