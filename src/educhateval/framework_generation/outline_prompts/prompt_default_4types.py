# prompts to generate default framework in py format to match outline requirements
# this template is relying on the categories/intentions suggested in this paper: https://doi.org/10.1016/j.ipm.2021.102838
# in addition a fourth category is added to the original paper, that is: statement

prompt_dict = {
    # CLARIFICATION
    "Clarification": """<|im_start|>system
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
I don't understand, can you make it simpler?
<|im_end|>
<|im_start|>assistant""",
    # SMALL TALK
    "Small Talk": """<|im_start|>system
You generate a short small talk sentence suitable in an casual setting. You never repeat yourself.
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
<|im_start|>assistant""",
    # QUESTION
    "Question": """<|im_start|>system
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
<|im_start|>assistant""",
    # STATEMENT
"Statement": """<|im_start|>system
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
<|im_start|>assistant"""
}
