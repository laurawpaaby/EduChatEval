### 1. THIS FIRST BIT IS FOR THE FIRST STEP OF THE PIPELINE: FRAMEWORK GENERATION
### THIS CREATES THE SYNTHETIC DATASET USED FOR TRAINING THE CLASSIFIER AND IS THE FOUNDATION FOR THE ANALYSIS

## LAURA TO DO: ADD EXPLAINING ERRORS IF INPUTS ARE NOT AS EXPECTED OR MISSING :D <333     
from typing import Union, Optional
import pandas as pd


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
                           prompt_path: str = None,
                           prompt_dict_input: dict = None,
                           num_samples: int = 500,
                           api_url: str = "http://localhost:1234/v1/completions",
                           json_out: str = None, 
                           csv_out: str = None):
        """
        Load prompt dict and generate synthetic labeled dataset.
        Returns a pandas DataFrame.
        """
        # use the prompt path to load the prompt dictionary
        #prompt_dict = load_prompt_dict_from_py(prompt_path) NOW DONE DIRECTLY BELOW IF CHOSEN OVER DICT 

        # generate the dataset using func 
        df = synthesize_dataset(
            prompt_dict=prompt_dict_input,
            prompt_path=prompt_path,
            model_name=self.model_name,
            num_samples=num_samples,
            api_url=api_url,
            json_out=json_out,
            csv_out=csv_out
        )

        return df
    

    #### 2. function to quality check the dataset
    from src.classification_utils import (load_tokenizer, load_and_prepare_dataset, tokenize_dataset, 
    train_model, save_model_and_tokenizer)

    from src.framework_generation.train_tinylabel_classifier import filter_synthesized_data

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
from typing import Optional
import pandas as pd
from pathlib import Path

from src.dialogue_generation.simulate_dialogue import simulate_conversation
from src.dialogue_generation.txt_llm_inputs.prompt_loader import load_prompts_and_seed  
from src.dialogue_generation.models.wrap_huggingface import ChatHF
from src.dialogue_generation.models.wrap_micr import ChatMLX


class DialogueSimulator:
    """
    Class to simulate a multi-turn dialogue between a student and tutor agent.
    Outputs structured data as a DataFrame or optional CSV.
    """

    def __init__(
        self,
        backend: str = "hf",
        model_id: str = "gpt2",
        sampling_params: Optional[dict] = None
    ):
        if backend == "hf":
            self.model = ChatHF(model_id=model_id, sampling_params=sampling_params or {
                "temperature": 0.9, "top_p": 0.9, "top_k": 50
            })
        elif backend == "mlx":
            self.model = ChatMLX(model_id=model_id, sampling_params=sampling_params or {
                "temp": 0.9, "top_p": 0.9, "top_k": 40
            })
        else:
            raise ValueError("Unsupported backend")

        self.model.load()

    def simulate_dialogue(
        self,
        mode: str = "general_course_exploration",
        turns: int = 5,
        log_dir: Optional[Path] = None,
        save_csv_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Simulate the conversation and return as DataFrame. Optionally save to CSV and log.
        """
        system_prompts = load_prompts_and_seed(mode)
        df = simulate_conversation(
            model=self.model,
            system_prompts=system_prompts,
            turns=turns,
            log_dir=log_dir,
            save_csv_path=save_csv_path
        )
        return df



###### 3. NOW DIALOGUE LOGGER FOR DIRECT INTERACTIONS WITH LLMS FROM LM STUDIO
from src.dialogue_wrapper.dia_wrapper_funcs import DialogueLogger
__all__ = ["DialogueLogger"] # all at once cause already class in that script - could be made here directly, but I prefer the seperation




###### 4. NOW LETS ADD THE CLASSIFIER FOR THE DIALOGUE DATA !!! :DDD
from src.classification_utils import (load_tokenizer, load_and_prepare_dataset, tokenize_dataset, 
    train_model, save_model_and_tokenizer)
from src.dialogue_classification.train_classifier import predict_annotated_dataset 

class PredictLabels:
    """
    Wrapper class for training a classifier and using it to annotate a new dataset.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = load_tokenizer(model_name)

    def run_pipeline(
        self,
        train_data: Union[str, pd.DataFrame],
        new_data: Union[str, pd.DataFrame],
        # columns in the training data
        text_column: str = "text",
        label_column: str = "category",
        # text coliumn in the new data should it have a different name than text_column
        new_text_column: Optional[str] = None,
        split_ratio: float = 0.2,
        training_params: list = [0.01, 'cross_entropy', 5e-5, 8, 8, 4, 0.01],
        tuning: bool = False,
        tuning_params: Optional[dict] = None,
        model_save_path: Optional[str] = None,
        prediction_save_path: Optional[str] = None) -> pd.DataFrame:

        """
        Trains classifier and returns annotated DataFrame.
        """

        dataset_dict, label2id = load_and_prepare_dataset(train_data, text_column, label_column, split_ratio)
        tokenized = tokenize_dataset(dataset_dict, self.tokenizer)

        model, trainer = train_model(tokenized, self.model_name, len(label2id), training_params, tuning, tuning_params)

        if model_save_path:
            save_model_and_tokenizer(model, self.tokenizer, model_save_path)

        df_annotated = predict_annotated_dataset(
            new_data=new_data,
            model=model,
            text_column=new_text_column,
            tokenizer=self.tokenizer,
            label2id=label2id,
            save_path=prediction_save_path
        )

        return df_annotated
