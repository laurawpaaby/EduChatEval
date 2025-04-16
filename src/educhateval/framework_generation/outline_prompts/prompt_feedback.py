#### prompt for feedback generation in py format to match outline requirements
# this template is relying on the feedback categories suggested in this paper: RASMUS PAPER SOURCS

prompt_dict = {
    "Feedback": """<|im_start|>system
You generate a one sentence feedback message.
It should be polite, concise, and appropriate for an educational setting. They should be constructive and simulate a teacher's feedback to a student.
<|im_end|>

<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>assistant
Whenever students lose focus I try to engage them asking them relevant questions.
<|im_end|>
<|im_start|>user
Create another feedback sentence.
<|im_end|>
<|im_start|>assistant
Your report covered the main points clearly and was easy to follow.
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>assistant
You've done an excellent job organizing the content in an appealing way!
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>assistant
Providing context for your points helped greatly in understanding your reasoning.
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>assistant
What do you want to achieve through this?
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>assistant
In the final conclusion you manage to wrap up the content nicely. 
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>user
Create a feedback sentence.
<|im_end|>
<|im_start|>assistant
Your use of rhetorical devices effectively keeps the reader's attention.
<|im_end|>
<|im_start|>assistant""",
}
