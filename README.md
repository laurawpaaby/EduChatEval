# chat interact wrap eduact :D 
something about this being the repo for my thesis with a niiice outline over how its structured: 
temporary picture of how it may look - mainly for my own overview:

![flowchart](flowchart.png)

### something about how to run and why 


# Documentation
links to website 

## Integration 
huggingface, lm studio etc use cases only

## Instalation
the goal - gotta figure out a name here:
``` bash
pip install chat_wrap_edu_pack 
```

## Usage 
well first of all because it is a package and not a script to run from the terminal, we see no parser = argparse...
so it is build to be a python package where each function are callable - fingers crossed it will be :D 

``` python
from chat_wrap_edu_pack install generate 
```

# Acknowdledgement 



# Complete overview:
``` 
├── data/                                  
│   ├── generated_dialogue_data/           # Generated dialogue samples
│   ├── generated_tuning_data/             # Generated framework data for fine-tuning 
│   ├── logged_dialogue_data/              # Logged real dialogue data
│   ├── tiny_labeled_data.csv              # Manual true data (needs folder)
│   └── tiny_labeled_feedback.csv          # -----..------- 
│
├── Models/                                # Folder for trained models and checkpoints (ignored)
│
├── src/                                   # Main source code for all components
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
│   ├── dialogue_wrapper/
│   │   ├── dia_wrapper_funcs.py/         # Wraps and manages userinteraction with llm 
│   ├── framework_generation/            
│   │   ├── outline_prompts/              # Prompt templates for outlines
│   │   ├── outline_synth_LMSRIPT.py      # Synthetic outline generation pipeline
│   │   └── train_tinylabel_classifier.py # Training classifier on manually made true data
│
├── mypythonpackage.py                    # Entrypoint script for the package 
├── testing_mypython_package.ipynb        # notebook for testing and demonstration  
├── .python-version                       # Python version file for (Poetry)
├── poetry.lock                           # Locked dependency versions (Poetry)
├── pyproject.toml                        # Main project config and dependencies
``` 