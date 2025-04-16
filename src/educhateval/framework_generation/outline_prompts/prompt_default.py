# prompts to generate default framework in py format to match outline requirements
# this template is relying on the categories/intentions suggested in this paper: https://doi.org/10.1016/j.ipm.2021.102838

prompt_dict = {
    "Clarification": """<|im_start|>system
You generate a conversational sentence that seeks clarification or context. 
It should be polite, concise, and appropriate for an educational setting.
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
    "Small Talk": """<|im_start|>system
You generate a short small talk sentence suitable in an casual setting.
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
    "Question": """<|im_start|>system
You generate a factual or thoughtful question that can be used in a conversation or educational setting.
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
}
