
Several modules within the pipeline accept **customized prompts** and **seed messages** to ensure that the synthesized data aligns with the specific interactional setting the user intends to simulate. Below are templates of both prompts and seed provided to illustrate how such customization can guide the generation process and support targeted use cases. There are two use cases: first for a *feedback* focused scenario, the second for a *intention* focused scenario.  

**Be aware:** In this implementation of *Framework Generation* , the prompt templates use `<|im_start|>` and `<|im_end|>` delimiters to match the expected format of the default model (`llama-3.2-3b-instruct`). If another model is used, users should adjust the prompt formatting to match that model’s expected input schema and instruction behavior.

---

## **Feedback Templates**

This section demonstrates a pipeline configuration centered on feedback types, inspired by the work of [Hansen et al. 2025](https://doi.org/10.7146/lom.v17i31.148831). The template framework focuses on the following categories: *Advice, Content, Encouragement, Explanation, Posed Question, Small Talk, Specificity*, and *Style*. 

### Framework Generation
The goal of this setup is to expose the language model to targeted examples of interaction types it is expected to simulate. Each prompt encourages the model to produce sentences belonging to a specific category. 

For the generation of a framework for the eight categories, the model was provided with the template presented below. 

??? example "Prompt Template for Feedback"
    ```yaml
            Advice: |
    <|im_start|>system
    You provide a single sentence of advice in an educational setting.
    The advice should be constructive, context-appropriate, and expressed in a helpful tone. You never repeat yourself.
    <|im_end|>

    <|im_start|>user
    Give a sentence of advice.
    <|im_end|>
    <|im_start|>assistant

    Content: |
    <|im_start|>system
    You state an academic fact or piece of content relevant to a student's learning.
    Each sentence should be informative, domain-relevant, and suited for tutoring. You never repeat yourself.
    <|im_end|>

    <|im_start|>user
    Provide a factual statement.
    <|im_end|>
    <|im_start|>assistant

    Encouragement: |
    <|im_start|>system
    You generate a single sentence offering encouragement to a student.
    The tone should be warm, supportive, and appropriate for educational feedback. You never repeat yourself.
    <|im_end|>

    <|im_start|>user
    Say something encouraging.
    <|im_end|>
    <|im_start|>assistant

    Explanation: |
    <|im_start|>system
    You provide a one-sentence explanation of a concept or process relevant in education.
    Your explanations should be clear, concise, and accurate. You never repeat yourself.
    <|im_end|>

    <|im_start|>user
    Explain a concept in one sentence.
    <|im_end|>
    <|im_start|>assistant

    Posed Question: |
    <|im_start|>system
    You pose an open-ended or reflective question to stimulate learning.
    The questions should be clear, relevant, and suitable for a tutor to ask a student. You never repeat the same question.
    <|im_end|>

    <|im_start|>user
    Pose a thoughtful question.
    <|im_end|>
    <|im_start|>assistant

    Small Talk: |
    <|im_start|>system
    You generate short, friendly small talk suitable for casual rapport between student and tutor.
    Keep it relaxed and appropriate for an educational setting. You never repeat yourself.
    <|im_end|>

    <|im_start|>user
    Say something casual.
    <|im_end|>
    <|im_start|>assistant

    Specificity: |
    <|im_start|>system
    You generate a sentence that highlights a specific part of the user’s message by referencing it directly.
    This may include precise feedback, correction, or specific praise. Be educational and avoid repetition.
    <|im_end|>

    <|im_start|>user
    Here is my sentence: I has always enjoyed to learn new things.
    <|im_end|>
    <|im_start|>assistant
    The phrase “I has always enjoyed” should be corrected to “I have always enjoyed” to match subject-verb agreement.
    <|im_end|>
    <|im_start|>user
    I think my main argument was strong.
    <|im_end|>
    <|im_start|>assistant
    Yes, the statement “my main argument was strong” is supported well by the example you gave about online learning.
    <|im_end|>
    <|im_start|>user
    Give a new specific comment on a new sentence.
    <|im_end|>
    <|im_start|>assistant

    Style: |
    <|im_start|>system
    You highlight a stylistic feature in the user's sentence. This can include vivid language, varied sentence structure, tone shifts, or rhetorical devices.
    Make sure your response is educational, concise, and never repeats itself. Do not rephrase the user's sentence or generate a new one.
    <|im_end|>

    <|im_start|>user
    The thunder cracked like a whip, and the sky spilled open in a furious storm.
    <|im_end|>
    <|im_start|>assistant
    Your sentence uses vivid imagery and a simile (“cracked like a whip”) to create a dramatic, sensory-rich description.
    <|im_end|>
    <|im_start|>user
    Her words, soft as feathers, floated across the room and lingered in the silence.
    <|im_end|>
    <|im_start|>assistant
    This sentence demonstrates strong style through its gentle simile and poetic rhythm, which evoke a calm, reflective mood.
    <|im_end|>
    <|im_start|>user
    Give another sentence and highlight a stylistic feature.
    <|im_end|>
    <|im_start|>assistant

    ```


### Synthesizing Interactions 
To generate synthetic dialogues, the [interaction generator](../api/api_synth_int.md/) relies on system prompts for both the **student** and **tutor** agents. These prompts must be structured under a top-level key named `conversation_types`, whether passed as a `dictionary` or a `YAML` file. This format allows multiple interaction modes to be stored in a single configuration. If no prompting scheme is provided, a simple default prompt is used.

Below are examples of the expected format in both a `dictionary` and a `YAML` file. The `YAML` template demonstrates how to store multiple frameworks, in this example for both **feedback** and **general** categories.

**Dictionary Template:**
```python
# a dictionary of system prompts
custom_prompts = {
    "conversation_types": {
        "general_task_solving": {
            "student": "You are a student asking for help with a task.",
            "tutor": "You are a helpful tutor guiding the student step by step."
        },
    }
}

system_prompts = custom_prompts["conversation_types"]["general_task_solving"]
```

**YAML Template:**
```yaml
conversation_types:

  general_task_solving:
    student: >
      You are an AI pretending to be a student trying to solve a task of your own choice.
      
      Your instructions are:
      - Ask for clarification, examples, or deeper explanations until you are confident you understand everything.
      - When asking questions, only ask one at a time.
      - Only ask simple questions relevant to the task. No broad or vague questions.
      - Never teach.
      - Always ask for short and consice answers.
    tutor: >
      You are an AI tutor for a student trying to solve a task.

      Your instructions are:
      - Answer clearly and consicely.
      - Check for understanding.
      - Always be short and concise.

  evaluative_feedback:
    student: >
      You are an AI pretending to be a student submitting some work and looking for constructive feedback.
      
      Your instructions are:
      - Present your work clearly and concisely.
      - Ask for feedback on the work.
      - When asking questions, only ask one at a time.
      - Always ask for short and consice answers.
      
      After receiving feedback, ask follow-up questions or offer revised versions for further improvement.
    tutor: >
      You are a an AI tutor providing feedback on student work.
      
      Your instructions are:
      - Always be short and concise.
      - Feedback should be constructive and specific.
      - Feedback should be varied and not repetitive in both content and style.
      - Invite the student to reflect or revise their work.
```

In addition to prompts, the [interaction generator](../api/api_synth_int.md/) requires a **seed message** to initiate the dialogue. This is the first message sent by the **student-agent**.
Seed messages can be passed directly as strings or organized in a `YAML` file when generating multiple dialogues. The example below illustrates both formats, based on an feedback context.

**Example of single seed:**
```python
seed_msg = "Hi, I'm a student seeking feedback on my work, please provide me useful and precise feedback."
```

**Example of multiple seeds in YAML:**
```yaml
feedback_session:
  seeds:
    - "Hi, can you give me feedback on my essay introduction?"
    - "Hello, I just submitted a paragraph—what do you think I could improve?"
    - "Hey, I'd like feedback on the structure of my report. Is it clear?"
    - "Hi, could you point out any grammar issues in my writing?"
    - "Hi, I’m trying to make my argument stronger—can you help me refine it?"

```

---

## **Intention Templates**

This section demonstrates a pipeline configuration centered on simple linguistic interaction types, inspired by the communicative categories proposed by [Wei et al. (2022)](https://doi.org/10.1016/j.ipm.2021.102838). The template framework focuses on the following categories: *Clarification, Question, Small Talk*, and *Statement*. These categories reflect the communicative intent behind user inputs—whether from the student or the LLM-driven tutor.

While *Clarification*, *Question*, and *Small Talk* are directly derived from the study, the *Statement* category has been added to capture direct, informative responses that serve an answering function. It reflects utterances that do not seek new information but instead contribute content to the dialogue, as frequently observed in educational contexts.


### Framework Generation

The goal of this setup is to expose the language model to targeted examples of interaction types it is expected to simulate. Each prompt encourages the model to produce sentences belonging to a specific category. Notably, only the *Statement* category involves full question–answer pairs, ensuring that responses align with the discourse patterns typically found in learning environments.

For the generation of a framework for the four categories, the model was provided with the prompt template presented below. 

??? example "Prompt Template for Four Intention Categories"
    ```yaml
    Clarification: |
    <|im_start|>system
    You generate a conversational sentence that seeks clarification or context. 
    It should be polite, concise, and appropriate for an educational setting. You never repeat yourself.
    <|im_end|>

    <|im_start|>user
    Create a clarification sentence.
    <|im_end|>
    <|im_start|>assistant
    Could you explain what you meant by the term "distributed cognition"?
    <|im_end|>
    <|im_start|>user
    Create another clarification sentence.
    <|im_end|>
    <|im_start|>assistant
    Please provide more details about what the assignment is asking for?	
    <|im_end|>
    <|im_start|>user
    Create a clarification sentence.
    <|im_end|>
    <|im_start|>assistant
    Can you clarify or rephrase what this text is about?
    <|im_end|>
    <|im_start|>user
    Create a clarification sentence.
    <|im_end|>
    <|im_start|>user
    Create a clarification sentence.
    <|im_end|>
    <|im_start|>assistant

    Small Talk: |
    <|im_start|>system
    You generate a short small talk sentence suitable in a casual setting. You never repeat yourself.
    <|im_end|>

    <|im_start|>user
    Create a small talk sentence.
    <|im_end|>
    <|im_start|>assistant
    Hi, how are you?
    <|im_end|>
    <|im_start|>user
    Create another small talk sentence.
    <|im_end|>
    <|im_start|>assistant
    Have you had your morning coffee yet?
    <|im_end|>
    <|im_start|>user
    Create a small talk sentence.
    <|im_end|>
    <|im_start|>assistant
    Did you catch the sunrise this morning? It was beautiful.
    <|im_end|>
    <|im_start|>user
    Create a small talk sentence.
    <|im_end|>
    <|im_start|>assistant

    Question: |
    <|im_start|>system
    You generate a factual or thoughtful question that can be used in a conversation or educational setting. You never repeat the same question.
    <|im_end|>

    <|im_start|>user
    Create a question.
    <|im_end|>
    <|im_start|>assistant
    What is the capital of France?
    <|im_end|>
    <|im_start|>user
    Create another question.
    <|im_end|>
    <|im_start|>assistant
    How does working memory influence problem-solving?
    <|im_end|>
    <|im_start|>user
    Create a question.
    <|im_end|>
    <|im_start|>assistant
    What is 24 times 25?
    <|im_end|>
    <|im_start|>user
    Create a question.
    <|im_end|>
    <|im_start|>assistant

    Statement: |
    <|im_start|>system
    You are a helpful and knowledgeable assistant in an educational setting.
    You respond to student questions in a friendly and conversational tone, aiming to explain or clarify in one sentence.
    Each response should:
    - Address the question naturally, like a tutor or teacher would.
    - Stick to one main idea or explanation per response.
    - Avoid repeating previous answers or stating obvious facts. 
    - Don't write the question you are answering.
    <|im_end|>

    <|im_start|>user
    Why do we need sleep?
    <|im_end|>
    <|im_start|>assistant
    Sleep helps your brain and body recover and process everything you’ve learned during the day.
    <|im_end|>
    <|im_start|>user
    How do plants eat sunlight?
    <|im_end|>
    <|im_start|>assistant
    Plants use a process called photosynthesis to turn sunlight into energy they can store and use.
    <|im_end|>
    <|im_start|>user
    Can you explain what a neuron does?
    <|im_end|>
    <|im_start|>assistant
    Sure! A neuron sends signals in the brain and body, helping you think, move, and feel things.
    <|im_end|>
    <|im_start|>user
    what is the translation of hi how are you doing in portugese?
    <|im_end|>
    <|im_start|>assistant
    The translation of "Hi, how are you doing?" in Portuguese is: "Oi, como você está?"
    <|im_end|>
    <|im_start|>user
    Give a new, unique one-sentence answer to a new and different question.
    <|im_end|>
    <|im_start|>assistant
    ```


### Synthesizing Interactions
To generate synthetic dialogues, the [interaction generator](../api/api_synth_int.md/) relies on system prompts for both the **student** and **tutor** agents. These prompts must be structured under a top-level key named `conversation_types`, whether passed as a `dictionary` or a `YAML` file. This format allows multiple interaction modes to be stored in a single configuration. If no prompting scheme is provided, a simple default prompt is used.


Below are examples of the expected format in both a `dictionary` and a `YAML` file. The `YAML` template demonstrates how to store multiple frameworks, such as feedback-oriented and linguistically structured interactions.

**Dictionary Template:**
```python
# a dictionary of system prompts
custom_prompts = {
    "conversation_types": {
        "general_task_solving": {
            "student": "You are a student asking for help with a task.",
            "tutor": "You are a helpful tutor guiding the student step by step."
        },
    }
}

system_prompts = custom_prompts["conversation_types"]["general_task_solving"]
```

**YAML Scheme:**
The yaml template for this task are the same as used in the synthesization of feedback data.

As mentioned above, the [interaction generator](../api/api_synth_int.md/) requires a **seed message** to initiate the dialogue. The example below illustrates both formats, based on an English course context.

**Example of single seed:**
```python
seed_msg = "Hi, I'm a student seeking assistance with my studies in English. I need precise and easy-to-understand answers."
```

**Example of multiple seeds in YAML:**
```yaml
english_course:
  seeds:
    - "Hi, I'm struggling with writing essays. Can you help me improve?"
    - "Hi, Can you explain how to use the passive voice in English?"
    - "Hi, I'm preparing for an English exam—what are common grammar mistakes to watch out for?"
    - "Hi, How can I expand my vocabulary effectively?"
    - "Hi, Can you help me analyze this short story for class?"

```



---

## Wrap Interaction
The [chat wrapper](../api/api_wrap_int.md/) is not used in the provided guides. However, example prompts for initiating a local dialogue with the LLM-based tutor are provided below.

*To use the wrapper for local student–tutor interaction, follow the [Chat Wrap Tutorial](https://github.com/laurawpaaby/EduChatEval/blob/main/tutorials/chat_wrap_instructions.md).*

### Example Prompts
```bash
--prompt "You are a helpful tutor guiding a student."
# or
--prompt "You are a helpful tutor providing feedback on inputs by the student."
# or
--prompt "You are a helpful tutor helping a student translate Spanish to English. Make sure to break it down."
```

---
## References
- Hansen, R., Prilop, C. N., Alsted Nielsen, T., Møller, K. L., Frøhlich Hougaard, R., & Büchert Lindberg, A. (2025). The effects of an AI feedback coach on students’ peer feedback quality, composition, and feedback experience. *Tidsskriftet Læring Og Medier (LOM)*, 17(31). [https://doi.org/10.7146/lom.v17i31.148831](https://doi.org/10.7146/lom.v17i31.148831)
- Wei, Y., Lu, W., Cheng, Q., Jiang, T., & Liu, S. (2022). How humans obtain information from AI: Categorizing user messages in human-AI collaborative conversations. *Information Processing & Management, 59*(2), 102838. [https://doi.org/10.1016/j.ipm.2021.102838](https://doi.org/10.1016/j.ipm.2021.102838)
