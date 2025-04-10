# prompts to generate default framework in py format to match outline requirements
# this template is relying on the categories/intentions suggested in this paper: https://doi.org/10.1016/j.ipm.2021.102838
# in addition a fourth category is added to the original paper, that is: statement

prompt_dict = {
    # CLARIFICATION
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
    # SMALL TALK
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
    # QUESTION
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
    # STATEMENT
    "Statement": """<|im_start|>system
You are a helpful assistant that answers factual questions in a clear, conversational tone. 
Each of your responses should answer the user's question directly with accurate information.
You only respond with a single sentence that could realistically come from an AI assistant like ChatGPT.
<|im_end|>

<|im_start|>user
What is the tallest mountain on Earth?
<|im_end|>
<|im_start|>assistant
Mount Everest is the tallest mountain on Earth, standing at 8,848 meters above sea level.
<|im_end|>
<|im_start|>user
Why do leaves change color in the fall?
<|im_end|>
<|im_start|>assistant
Leaves change color in the fall because chlorophyll breaks down, revealing pigments like carotenoids and anthocyanins.
<|im_end|>
<|im_start|>user
How does photosynthesis work?
<|im_end|>
<|im_start|>assistant
Photosynthesis is the process where plants use sunlight, carbon dioxide, and water to produce energy and oxygen.
<|im_end|>
<|im_start|>user
What is the boiling point of water?
<|im_end|>
<|im_start|>assistant
Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at sea level.
<|im_end|>
<|im_start|>user
What is a black hole?
<|im_end|>
<|im_start|>assistant
A black hole is a region in space where gravity is so strong that nothing, not even light, can escape from it.
<|im_end|>
<|im_start|>user
Give a one-sentence factual answer.
<|im_end|>
<|im_start|>assistant""",
}
