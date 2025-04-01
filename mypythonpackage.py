### 1. THIS FIRST BIT IS FOR THE FIRST STEP OF THE PIPELINE: FRAMEWORK GENERATION
### THIS CREATES THE SYNTHETIC DATASET USED FOR TRAINING THE CLASSIFIER AND IS THE FOUNDATION FOR THE ANALYSIS

## LAURA TO DO: ADD EXPLAINING ERRORS IF INPUTS ARE NOT AS EXPECTED OR MISSING :D <333     

# Framework Generator Modules:
from framework_generation.train_tinylabel_classifier import filter_synthesized_data, load_and_prepare_dataset, load_tokenizer, save_model_and_tokenizer, tokenize_dataset, train_model
from src.framework_generation.outline_synth_LMSRIPT import load_prompt_dict_from_py, synthesize_dataset


class FrameworkGenerator:
    """
    High-level interface for generating synthetic frameworks using prompts and a local model.
    """

    def __init__(self, model_name: str = "llama-3.2-3b-instruct"):
        self.model_name = model_name

    #### 1. function to generate the raw dataset, not yet filtered and quality checked
    def generate_framework(self, 
                           prompt_path: str, 
                           num_samples: int = 1000, 
                           api_url: str = "http://localhost:1234/v1/completions",
                           json_out: str = None, 
                           csv_out: str = None):
        """
        Load prompt dict and generate synthetic labeled dataset.
        Returns a pandas DataFrame.
        """
        # use the prompt path to load the prompt dictionary
        prompt_dict = load_prompt_dict_from_py(prompt_path)

        # generate the dataset using func 
        df = synthesize_dataset(
            prompt_dict=prompt_dict,
            model_name=self.model_name,
            num_samples=num_samples,
            api_url=api_url,
            json_out=json_out,
            csv_out=csv_out
        )

        return df
    

    #### 2. function to quality check the dataset
    from typing import Union
    import pandas as pd
    from src.framework_generation.train_tinylabel_classifier import (load_tokenizer, load_and_prepare_dataset,tokenize_dataset,
        train_model, save_model_and_tokenizer, filter_synthesized_data)

    def filter_with_classifier(self,
                            train_data: Union[str, pd.DataFrame],
                            synth_data: Union[str, pd.DataFrame],
                            text_column: str = "text",
                            label_column: str = "category",
                            split_ratio: float = 0.2,
                            training_params: list = [0.01, 'cross_entropy', 5e-5, 8, 8, 4, 0.01],
                            tuning: bool = False,
                            tuning_params: dict = None,
                            model_save_path: str = None,
                            classifier_model_name: str = "distilbert-base-uncased",
                            filtered_save_path: str = None) -> pd.DataFrame:
        """
        Train a small classifier on labeled data and filter synthetic data based on prediction agreement.
        Accepts training and synthetic data as file paths or DataFrames.
        Returns the filtered high-quality dataset as a pandas DataFrame.
        """
        tokenizer = load_tokenizer(classifier_model_name)
        dataset_dict, label2id = load_and_prepare_dataset(train_data, text_column, label_column, split_ratio)
        tokenized = tokenize_dataset(dataset_dict, tokenizer)
        model, trainer = train_model(tokenized, classifier_model_name, len(label2id), training_params, tuning, tuning_params)

        trainer.evaluate()

        if model_save_path:
            save_model_and_tokenizer(model, tokenizer, model_save_path)

        df_filtered = filter_synthesized_data(
            synth_input=synth_data,
            model=model,
            tokenizer=tokenizer,
            label_column=label_column,
            save_path=filtered_save_path
        )

        return df_filtered
    

    


#### 2. NOW NEXT STEP SHOULD BE GENERATING THE SYNTHETIC DIALOGUE DATA ..........
class DialogueGenerator:
    """
    Generates synthetic dialogue data using a trained quality classifier.
    """

    #def __init__(
    #) 