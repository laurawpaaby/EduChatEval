![Screenshot](pics/frontpage.png)

This package offers a framework for researchers to log and classify interactions between students and LLM-based tutors in educational settings. It supports structured, objective evaluation through classification, simulation, and visualization utilities, and is designed for flexible use across tasks of any scale. The framework supports both researchers with pre-collected datasets and those operating in data-sparse contexts. It designed as a modular tool that can be integrated at any stage of the evaluation process.

The package is designed to:

- Synthesize a labeled classification framework using user-defined categories 
- Simulate multi-turn student–tutor dialogues via role-based prompting and structured seed messages
- Wrap direct student-tutors interaction with locally hosted LLMs through a terminal-based interface 
- Fine-tune and apply classification models to label conversational turns
- Visualize dialogue patterns with summary tables, frequency plots, temporal trends, and sequential dependencies

---

### User Guides and API 
To run the full pipeline, the package requires integration with LM Studio (for local model hosting) and the Hugging Face ecosystem. Step-by-step tutorials are provided in the [User Guide section](user_guides/userguide_intro.md/), covering setup, configuration, and usage across all modules—from data generation to classification and visualization.

Detailed information on the available classes, functions, and configuration options can be found in the [API reference](api/api_frame_gen.md/).

**Be aware**, that the package currently requires [`Python 3.12`](https://www.python.org/downloads/release/python-3120/) due to version constraints in core dependencies, particularly [`outlines`](https://github.com/dottxt-ai/outlines?tab=readme-ov-file#type-constraint).