![Screenshot](pics/frontpage.png)

This package offers a framework for researchers to map and quantify interactions between students and LLM-based tutors in educational settings. It supports structured, objective evaluation through classification, simulation, and visualization tools, and is designed for flexible use across tasks of any scale. The framework accommodates both researchers analyzing pre-collected, annotated data and those starting from scratch, providing modular support through each step of the evaluation process.

The package is designed to:

- Provide a customized framework for classification, evaluation, and fine-tuning
- Simulate student–tutor interactions using role-based prompts and seed messages when real data is unavailable
- Initiate an interface with locally hosted, open-source models (e.g., via LM Studio or Hugging Face)
- Log interactions in structured formats (JSON/CSV) for downstream analysis
- Train and applu classifiers to predict customized interaction classes and visualize patterns across conversations

---

### User Guides and API 
To run the full pipeline, the package requires integration with LM Studio (for local model hosting) and the Hugging Face ecosystem. Step-by-step tutorials are provided in the [User Guide section](user_guides/userguide_intro.md/), covering setup, configuration, and usage across all modules—from data generation to classification and visualization.

Detailed information on the available classes, functions, and configuration options can be found in the [API reference](api/api_frame_gen.md/).

**Be aware**, that the package currently requires [`Python 3.12`](https://www.python.org/downloads/release/python-3120/) due to version constraints in core dependencies, particularly [`outlines`](https://github.com/dottxt-ai/outlines?tab=readme-ov-file#type-constraint).