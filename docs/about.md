# About

### Project
This package was developed as part of a master's thesis in Cognitive Science at Aarhus University. The project explores how interactions between students and LLM-based tutors can be systematically analyzed using natural language processing, classification pipelines, and visualization techniques.

The core contribution is the enablement of quantitative evaluation of interaction-based learning by transforming student–tutor exchanges into structured, labeled data. Through classifier-based annotation and modular components for synthetic data generation, interaction simulation, and result visualization, the package enables a reproducible and data-driven approach to studying chatbot-assisted education. It is designed for flexibility across educational and experimental contexts, particularly where interaction analysis is needed beyond traditional self-report methods.

The full source code for the [EduChatEval package](https://pypi.org/project/educhateval/) is available on GitHub and is licensed under [MIT License](https://github.com/laurawpaaby/EduChatEval/blob/main/LICENSE)


### Contact
For questions or collaboration inquiries, feel free to reach out to [Laura W Paaby](mailto:laurawpaaby@gmail.com).


## Acknowledgements

This project builds on existing tools and ideas from the open-source community. While specific references are provided within the relevant scripts throughout the repository, the key sources of inspiration are also acknowledged here to highlight the contributions that have shaped the development of this package.

- *Constraint-Based Data Generation – Outlines Package*  
  [Willard, Brandon T. & Louf, Rémi (2023). *Efficient Guided Generation for LLMs.*](https://arxiv.org/abs/2307.09702)  
  Provided the foundation for constrained text generation used in the framework module.

- *Chat Interface and Wrapper – Textual* 
  [McGugan, W. (2024, Sep). *Anatomy of a Textual User Interface.*](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five)  
  Inspired the design and implementation of the terminal-based chat application.

- *Package Design Inspiration* 
  [Thea Rolskov Sloth & Astrid Sletten Rybner](https://github.com/DaDebias/genda-lens)  
  Their clean and modular implementation of Genda-Lens influenced the architectural design of *EduChatEval*.

- *Code Debugging and Conceptual Feedback*
  [Mina Almasi](https://pure.au.dk/portal/da/persons/mina%40cc.au.dk) and [Ross Deans Kristensen-McLachlan](https://pure.au.dk/portal/da/persons/rdkm%40cc.au.dk)  
  Contributed helpful debugging support and valuable conceptual input throughout the development process.
