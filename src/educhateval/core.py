# Overall library for the pipeline
from typing import Union, Optional, List
import pandas as pd
import numpy as np
import random
import torch

# Set seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


""" General Overview Over the Pipeline:
1. Generate a synthetic framework dataset using a local model and prompts.
2. Train a small classifier on a labeled dataset and filter the synthetic data based on prediction agreement.
3. Simulate a multi-turn dialogue between a student and tutor agent.
4. Log the conversation using a dialogue logger.
5. Train a classifier on the dialogue data and use it to annotate new datasets.
6. Save the annotated dataset and optionally the model.
7. Descriptive Results 
"""


### 1. Framework Generation: Synthesize an annotaded dataset using prompts and a local model

# Framework Generator Modules:
from educhateval.framework_generation.train_tinylabel_classifier import (
    filter_synthesized_data,
    load_and_prepare_dataset,
    load_tokenizer,
    save_model_and_tokenizer,
    tokenize_dataset,
    train_model,
)
from educhateval.framework_generation.outline_synth_LMSRIPT import (
    synthesize_dataset,
)

import warnings
warnings.filterwarnings("ignore", message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")


class FrameworkGenerator:
    """
    High-level interface for generating synthetic frameworks using prompts and a local model.
    """

    def __init__(self, model_name: str = "llama-3.2-3b-instruct", api_url: str = "http://localhost:1234/v1/completions"):
        self.model_name = model_name
        self.api_url = api_url

    #### 1. function to generate the raw dataset, not yet filtered and quality checked
    def generate_framework(
        self,
        prompt_path: str = None,
        prompt_dict_input: dict = None,
        num_samples: int = 500,
        json_out: str = None,
        csv_out: str = None,
        seed: int = 42
    ):
        """
        Load prompt dict and generate synthetic labeled dataset.
        Returns a pandas DataFrame.
        """

        set_seed(seed)

        df = synthesize_dataset(
            prompt_dict=prompt_dict_input,
            prompt_path=prompt_path,
            model_name=self.model_name,
            num_samples=num_samples,
            api_url=self.api_url,
            json_out=json_out,
            csv_out=csv_out,
        )

        return df

    #### 2. function to quality check the dataset
    from educhateval.classification_utils import (
        load_tokenizer,
        load_and_prepare_dataset,
        tokenize_dataset,
        train_model,
        save_model_and_tokenizer,
    )

    from educhateval.framework_generation.train_tinylabel_classifier import (
        filter_synthesized_data,
    )

    def filter_with_classifier(
        self,
        train_data: Union[str, pd.DataFrame],
        synth_data: Union[str, pd.DataFrame],
        text_column: str = "text",
        label_column: str = "category",
        split_ratio: float = 0.2,
        training_params: list = [0.01, "cross_entropy", 5e-5, 8, 8, 4, 0.01],
        tuning: bool = False,
        tuning_params: dict = None,
        model_save_path: str = None,
        classifier_model_name: str = "distilbert-base-uncased",
        filtered_save_path: str = None,
    ) -> pd.DataFrame:
        """
        Train a small classifier on labeled data and filter synthetic data based on prediction agreement.
        Accepts training and synthetic data as file paths or DataFrames.
        Returns the filtered high-quality dataset as a pandas DataFrame.
        """
        tokenizer = load_tokenizer(classifier_model_name)
        dataset_dict, label2id = load_and_prepare_dataset(
            train_data, text_column, label_column, split_ratio
        )
        tokenized = tokenize_dataset(dataset_dict, tokenizer)
        model, trainer = train_model(
            tokenized,
            classifier_model_name,
            len(label2id),
            training_params,
            tuning,
            tuning_params,
        )

        trainer.evaluate()

        if model_save_path:
            save_model_and_tokenizer(model, tokenizer, model_save_path)

        df_filtered = filter_synthesized_data(
            synth_input=synth_data,
            model=model,
            tokenizer=tokenizer,
            label_column=label_column,
            save_path=filtered_save_path,
        )

        return df_filtered


#### 2. NOW NEXT STEP IS GENERATING THE SYNTHETIC DIALOGUE DATA 
from typing import Optional
import pandas as pd
from pathlib import Path

from educhateval.dialogue_generation.simulate_dialogue import simulate_conversation
from educhateval.dialogue_generation.txt_llm_inputs.prompt_loader import load_prompts_and_seed
from educhateval.dialogue_generation.models.wrap_huggingface import ChatHF
from educhateval.dialogue_generation.models.wrap_micr import ChatMLX


class DialogueSimulator:
    """
    Class to simulate a multi-turn dialogue between a student and tutor agent.
    Outputs structured data as a DataFrame or optional CSV.
    """

    def __init__(
        self,
        backend: str = "hf",
        model_id: str = "gpt2",
        sampling_params: Optional[dict] = None,
    ):
        if backend == "hf":
            self.model = ChatHF(
                model_id=model_id,
                sampling_params=sampling_params
                or {"temperature": 0.9, "top_p": 0.9, "top_k": 50},
            )
        elif backend == "mlx":
            self.model = ChatMLX(
                model_id=model_id,
                sampling_params=sampling_params
                or {"temp": 0.9, "top_p": 0.9, "top_k": 40},
            )
        else:
            raise ValueError("Unsupported backend")

        self.model.load()

    def simulate_dialogue(
        self,
        mode: str = "general_course_exploration",
        turns: int = 5,
        seed_message_input: str = "Hi, I'm a student seeking assistance with my studies.",
        log_dir: Optional[Path] = None,
        save_csv_path: Optional[Path] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Simulate the conversation and return as DataFrame. Optionally save to CSV and log.
        """
        set_seed(seed)

        system_prompts = load_prompts_and_seed(mode)
        df = simulate_conversation(
            model=self.model,
            system_prompts=system_prompts,
            turns=turns,
            seed_message_input=seed_message_input,
            log_dir=log_dir,
            save_csv_path=save_csv_path,
        )


        print(f"\n Full dialogue stored in DataFrame: use the returned object or view as `df`")
        return df


###### 3. NOW DIALOGUE LOGGER FOR DIRECT INTERACTIONS WITH LLMS FROM LM STUDIO
# actually, this is not saved in the package, as it is used from the chat_ui.py func instead. Should be deleted.

from pathlib import Path
from educhateval.dialogue_generation.chat import ChatMessage, ChatHistory
from educhateval.dialogue_wrapper.app_lmstudio import ChatLMStudio, ChatApp  

class ChatWrap:
    
    """
    A wrapper class for launching the Textual chat interface
    with an LM Studio-backed language model.
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:1234/v1/chat/completions",
        model_name: str = "llama-3.2-3b-instruct",
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful tutor guiding a student. Answer short and concisely.",
        save_dir: Path = Path("data/logged_dialogue_data"),
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.save_dir = save_dir

        # Initialize model and conversation history
        self.model = ChatLMStudio(api_url=self.api_url, model_name=self.model_name, temperature=self.temperature)
        self.chat_history = ChatHistory(
            messages=[ChatMessage(role="system", content=self.system_prompt)]
        )

    def run(self):
        """Launch the Textual app."""
        app = ChatApp(
            model=self.model,
            chat_history=self.chat_history,
            chat_messages_dir=self.save_dir,
        )
        app.run()





###### 4. NOW LETS ADD THE CLASSIFIER FOR THE DIALOGUE DATA !!! :DDD
from educhateval.classification_utils import (
    load_tokenizer,
    load_and_prepare_dataset,
    tokenize_dataset,
    train_model,
    save_model_and_tokenizer,
)
from educhateval.dialogue_classification.train_classifier import predict_annotated_dataset


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
        # columns to classify in the new data
        columns_to_classify: Optional[Union[str, List[str]]] = None,
        split_ratio: float = 0.2,
        training_params: list = [0.01, "cross_entropy", 5e-5, 8, 8, 4, 0.01],
        tuning: bool = False,
        tuning_params: Optional[dict] = None,
        model_save_path: Optional[str] = None,
        prediction_save_path: Optional[str] = None,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Trains classifier and returns annotated DataFrame.
        If columns_to_classify is None, text_column is used for predictions.
        """
        set_seed(seed)

        dataset_dict, label2id = load_and_prepare_dataset(
            train_data, text_column, label_column, split_ratio
        )
        tokenized = tokenize_dataset(dataset_dict, self.tokenizer)

        model, trainer = train_model(
            tokenized,
            self.model_name,
            len(label2id),
            training_params,
            tuning,
            tuning_params,
        )

        if model_save_path:
            save_model_and_tokenizer(model, self.tokenizer, model_save_path)

        # Default to using the training text_column if no specific columns_to_classify provided
        if columns_to_classify is None:
            columns_to_classify = text_column

        df_annotated = predict_annotated_dataset(
            new_data=new_data,
            model=model,
            text_columns=columns_to_classify,  # Adjusted to handle list of columns
            tokenizer=self.tokenizer,
            label2id=label2id,
            save_path=prediction_save_path,
        )

        return df_annotated



### 5. Visualization and Analysis ####
from educhateval.descriptive_results.display_results import (
    plot_predicted_categories,
    plot_category_bars,
    create_prediction_summary_table,
    plot_previous_turn_distribution
)

class Visualizer:
    """
    Visualization class for analyzing predicted dialogue labels.
    Wraps existing plotting and summary functions from display_result.py.

    Parameters
    ----------
    df : pd.DataFrame
        The annotated dataframe containing predicted label columns.
    student_col : str, optional
        Name of the column containing student message predictions.
    tutor_col : str, optional
        Name of the column containing tutor message predictions.

    Other keyword arguments (**kwargs) are passed through to the internal plotting functions.
    """

    def plot_turn_trends(self, df, student_col=None, tutor_col=None, **kwargs):
        """Wrapper for turn-based category line plot."""
        return plot_predicted_categories(df, student_col=student_col, tutor_col=tutor_col, **kwargs)

    def plot_category_bars(self, df, student_col=None, tutor_col=None, **kwargs):
        """Wrapper for grouped barplot of predicted categories."""
        return plot_category_bars(df, student_col=student_col, tutor_col=tutor_col, **kwargs)

    def create_summary_table(self, df, student_col=None, tutor_col=None):
        """Wrapper for generating prediction summary table."""
        return create_prediction_summary_table(df, student_col=student_col, tutor_col=tutor_col)

    def plot_history_interaction(self, df, focus_agent='student', **kwargs):
        """Wrapper for barplot showing category transitions from previous turn."""
        return plot_previous_turn_distribution(df, focus_agent, **kwargs)
