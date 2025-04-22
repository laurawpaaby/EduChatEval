# 🧠 EduChatEval

_A structured pipeline and Python package for evaluating interactive LLM tutor systems in educational settings._

---

## 🚀 Overview

This repository supports my master’s thesis, exploring how large language models can simulate and support human-like educational dialogues.

The package is designed to:

- Simulate student–tutor dialogues with role-based prompts if no real data is available 
- Integrate local + open-source models (e.g., LM Studio, Hugging Face)
- Log interactions (JSON/CSV) for analysis
- Provide a framework for classification, evaluation, and fine-tuning

Here’s a rough overview of the system architecture:

![flowchart](new_flowchart.png)

---

## ⚙️ Installation

```bash
pip install educhateval
```


## 🤗 Integration 
- 🦙 LM Studio (local LLM inference)
- 🖍️ Outlines
- 🤗 Transformers
- 🧪 Optuna for experimental tuning



## 📖 Documentation

| **Documentation** | **Description** |
|-------------------|-----------------|
| 📚 [User Guide](https://laurawpaaby.github.io/EduChatEval/user-guide/) | Instructions on how to run simulations and analyze dialogue logs |
| 💡 [Prompt Templates](https://your-docs-site.com/api) | Overview of system prompts, role behaviors, and instructional strategies |
| 🧠 [API References](https://your-docs-site.com/api) | Full reference for the `educhateval` API: classes, methods, and usage |
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