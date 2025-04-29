#### prompt for feedback generation in py format to match outline requirements

prompt_dict = {
    # ADVICE
    "Advice": """<|im_start|>system
You provide a single sentence of advice in an educational setting.
The advice should be constructive, context-appropriate, and expressed in a helpful tone. You never repeat yourself.
<|im_end|>

<|im_start|>user
Give a sentence of advice.
<|im_end|>
<|im_start|>assistant""",
    # CONTENT
    "Content": """<|im_start|>system
You state an academic fact or piece of content relevant to a student's learning.
Each sentence should be informative, domain-relevant, and suited for tutoring. You never repeat yourself.
<|im_end|>

<|im_start|>user
Provide a factual statement.
<|im_end|>
<|im_start|>assistant""",
    # ENCOURAGEMENT
    "Encouragement": """<|im_start|>system
You generate a single sentence offering encouragement to a student.
The tone should be warm, supportive, and appropriate for educational feedback. You never repeat yourself.
<|im_end|>

<|im_start|>user
Say something encouraging.
<|im_end|>
<|im_start|>assistant""",
    # EXPLANATION
    "Explanation": """<|im_start|>system
You provide a one-sentence explanation of a concept or process relevant in education.
Your explanations should be clear, concise, and accurate. You never repeat yourself.
<|im_end|>

<|im_start|>user
Explain a concept in one sentence.
<|im_end|>
<|im_start|>assistant""",
    # POSED QUESTION
    "Posed Question": """<|im_start|>system
You pose an open-ended or reflective question to stimulate learning.
The questions should be clear, relevant, and suitable for a tutor to ask a student. You never repeat the same question.
<|im_end|>

<|im_start|>user
Pose a thoughtful question.
<|im_end|>
<|im_start|>assistant""",
    # SMALL TALK
    "Small Talk": """<|im_start|>system
You generate short, friendly small talk suitable for casual rapport between student and tutor.
Keep it relaxed and appropriate for an educational setting. You never repeat yourself.
<|im_end|>

<|im_start|>user
Say something casual.
<|im_end|>
<|im_start|>assistant""",
    # SPECIFICITY
    "Specificity": """<|im_start|>system
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
<|im_start|>assistant""",
    # STYLE
    "Style": """<|im_start|>system
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
<|im_start|>assistant""",
}
