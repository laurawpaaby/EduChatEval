Functions can be used to generate any relevant classification framework given predefined categories. 
The generation relies on the constraining synthesization method provided by 

This folder contains the pipeline for generating labeled synthetic dialogue data that serves as frameworks in the later analysis. It uses language models served via LM Studio and the [`outlines`](https://github.com/dottxt-ai/outlines?tab=readme-ov-file#type-constraint ) library to ensure structured and constrained output.

---

## Contents

### `outline_synth_LMSRIPT.py`
Main script for generating synthetic text samples in a structured way.

- Loads a prompt file with category-to-prompt mappings.
- Sends generation requests to LM Studio.
- Saves results in both JSON and CSV formats.
- Automatically removes duplicates and unwanted meta-output.

### `outline_prompts/prompt_default.py`
Python file that stores prompts in a dictionary format:

```python
prompt_dict = {
    "Feedback": "<|im_start|>system ...",
    "Clarification": "<|im_start|>system ..."
}
```

## Quality 
The quality of the generated data can be tested and ensured by the method presented here: https://ojs.aaai.org/index.php/AAAI/article/view/6233  To do so a small dataset on the format: text, label csv. is required to serve as the 'true standard'
this is done by 

### Usage 

``` python
# Step 1 - Generate a framework
df = generator.generate_framework(
    prompt_path="src/framework_generation/outline_prompts/prompt_default.py", ### THIS NEEDS TO BE UPDATED TO ALSO TAKE IN A DIRECTORY DIRECTLY 
    num_samples=10,
    api_url="http://localhost:1234/v1/completions"
    #json_out="output/data.json",
    #csv_out="output/data.csv"
)

# Step 2 - filter by trained classifier 
filtered_df = generator.filter_with_classifier(
    train_data=df_train,
    synth_data=df_synth
)
``` 
