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
1. Generate a synthetic framework dataset using a local model and prompts and train a small classifier on a labeled dataset and filter the synthetic data based on prediction agreement.
2. Simulate a multi-turn dialogue between a student and tutor agent.
3. Train a classifier on the dialogue data and use it to annotate new datasets. Save the annotated dataset and optionally the model.
4. Visualize the descriptive results 
"""


### 1. Framework Generation: Synthesize an annotaded dataset using prompts and a local model

# Framework Generator Modules:
from educhateval.classification_utils import (
    load_tokenizer,
    load_and_prepare_dataset,
    tokenize_dataset,
    train_model,
    save_model_and_tokenizer,
    filter_synthesized_data,
)

from educhateval.framework_generation.outline_synth_LMSRIPT import (
    synthesize_dataset,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.",
)


class FrameworkGenerator:
    """
    High-level interface for generating synthetic datasets (frameworks) using prompts and local LLMs.

    Includes:
    - Synthetic data generation using instruction-tuned models.
    - Filtering of low-quality examples via classifier agreement with a small labeled dataset.
    """

    def __init__(
        self,
        model_name: str = "llama-3.2-3b-instruct",
        api_url: str = "http://localhost:1234/v1/completions",
    ):
        """
        Initialize the framework generator.

        Parameters:
        - model_name (str): Name of the generative model to use (e.g., llama-3.2-3b-instruct).
        - api_url (str): URL endpoint to communicate with the locally hosted LLM (e.g., LM Studio).
        """
        self.model_name = model_name
        self.api_url = api_url

    def generate_framework(
        self,
        prompt_path: str = None,
        prompt_dict_input: dict = None,
        num_samples: int = 500,
        json_out: str = None,
        csv_out: str = None,
        seed: int = 42,
        temperature: float = 0.85,
        top_p: float = 0.90,
    ) -> pd.DataFrame:
        """
        Generate a synthetic labeled dataset from prompts using a language model.

        Either `prompt_path` (path to .py file with `prompt_dict`) or `prompt_dict_input` must be provided.

        Parameters:
        - prompt_path (str): Path to a Python file containing a prompt dictionary.
        - prompt_dict_input (dict): Prompt dictionary directly provided.
        - num_samples (int): Number of samples to generate per category.
        - json_out (str): Optional path to save JSON output.
        - csv_out (str): Optional path to save CSV output.
        - seed (int): Random seed for reproducibility.

        Returns:
        - pd.DataFrame: Cleaned, labeled synthetic dataset.
        """
        if not prompt_path and not prompt_dict_input:
            raise ValueError(
                "You must provide either a prompt_path or prompt_dict_input."
            )

        set_seed(seed)

        df = synthesize_dataset(
            prompt_dict=prompt_dict_input,
            prompt_path=prompt_path,
            model_name=self.model_name,
            num_samples=num_samples,
            api_url=self.api_url,
            json_out=json_out,
            csv_out=csv_out,
            temperature=temperature,
            top_p=top_p,
        )

        return df

    #### 2. function to quality check the dataset
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
        Train a small classifier on real labeled data and use it to filter the synthetic dataset by agreement.

        Parameters:
        - train_data (str or pd.DataFrame): Path or DataFrame of small labeled training set.
        - synth_data (str or pd.DataFrame): Path or DataFrame of generated synthetic dataset.
        - text_column (str): Name of the text column.
        - label_column (str): Name of the label column.
        - split_ratio (float): Ratio for train/test split.
        - training_params (list): Training hyperparameters.
        - tuning (bool): Whether to perform hyperparameter tuning using Optuna.
        - tuning_params (dict): Optional tuning grid.
        - model_save_path (str): Optional path to save the classifier model.
        - classifier_model_name (str): HF model ID for the classifier.
        - filtered_save_path (str): Optional path to save filtered synthetic dataset.

        Returns:
        - pd.DataFrame: Filtered synthetic dataset based on classifier agreement.
        """
        if isinstance(train_data, pd.DataFrame) and train_data.empty:
            raise ValueError("Provided training DataFrame is empty.")
        if isinstance(synth_data, pd.DataFrame) and synth_data.empty:
            raise ValueError("Provided synthetic DataFrame is empty.")

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


#### 2. GENERATION OF SYNTHETIC DIALOGUE DATA
from typing import Optional
import pandas as pd
from pathlib import Path

from educhateval.dialogue_generation.simulate_dialogue import simulate_conversation
from educhateval.dialogue_generation.txt_llm_inputs.prompt_loader import (
    load_prompts_and_seed,
)
from educhateval.dialogue_generation.models.wrap_huggingface import ChatHF
from educhateval.dialogue_generation.models.wrap_micr import ChatMLX


class DialogueSimulator:
    """
    A modular simulator for generating multi-turn dialogues between a student and tutor agent using large language models.

    This class wraps backend-specific model interfaces and orchestrates the simulation of goal-driven conversations across various educational modes.
    It supports customizable sampling behavior and ensures reproducibility via global seeding. Outputs are returned as structured pandas DataFrames.

    Attributes:
        backend (str): Backend to use for inference. Options are "hf" (Hugging Face) or "mlx" (MLX).
        model_id (str): The identifier of the model to use, e.g., "gpt2" or an MLX-compatible model.
        sampling_params (Optional[dict]): Sampling hyperparameters such as temperature, top_p, or top_k.

    Methods:
        simulate_dialogue(...): Simulates a dialogue and returns it as a pandas DataFrame.
    """

    def __init__(
        self,
        backend: str = "mlx",
        model_id: str = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit",
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
        mode: str = "general_task_solving",
        turns: int = 5,
        seed_message_input: str = "Hi, I'm a student seeking assistance with my studies.",
        log_dir: Optional[Path] = None,
        save_csv_path: Optional[Path] = None,
        seed: int = 42,
        custom_prompt_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Simulates a multi-turn dialogue using either built-in or custom prompts.

        Args:
            mode: Mode key to select prompt pair (student/tutor).
            turns: Number of back-and-forth turns to simulate.
            seed_message_input: First message from the student.
            log_dir: Directory to save raw log (optional).
            save_csv_path: Path to save structured DataFrame (optional).
            seed: Random seed for reproducibility.
            custom_prompt_file: Optional path to custom YAML defining prompt modes.

        Returns:
            pd.DataFrame: Structured DataFrame of the conversation.
        """
        set_seed(seed)

        # system_prompts = load_prompts_and_seed(mode)

        df = simulate_conversation(
            model=self.model,
            turns=turns,
            seed_message_input=seed_message_input,
            log_dir=log_dir,
            save_csv_path=save_csv_path,
            custom_prompt_file=custom_prompt_file,
            mode=mode,
        )

        print(
            f"\n Full dialogue stored in DataFrame: use the returned object or view as `df`"
        )
        return df


###### (3). DIALOGUE LOGGER FOR DIRECT INTERACTIONS WITH LLMS FROM LM STUDIO
# This is saved as a function of the package and not as a class here. Find it in the chat_ui.py file.


###### 3. CLASSIFIER FOR THE DIALOGUE DATA
from educhateval.classification_utils import (
    load_tokenizer,
    load_and_prepare_dataset,
    tokenize_dataset,
    train_model,
    save_model_and_tokenizer,
)
from educhateval.dialogue_classification.train_classifier import (
    predict_annotated_dataset,
)


class PredictLabels:
    """
    A wrapper for training and applying a text classification model.

    This class streamlines the process of fine-tuning a transformer-based classifier on labeled data
    and applying the trained model to annotate new, unlabeled datasets. Supports both single and multi-column
    predictions and includes optional model saving and evaluation output.

    Required Args:
        model_name (str): Name of the pretrained Hugging Face model to fine-tune (default: "distilbert-base-uncased").

    Methods:
        run_pipeline(...): Trains the classifier and returns a DataFrame with predicted labels and confidence scores.
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
        seed: int = 42,
    ) -> pd.DataFrame:

        # Validate training data input
        if not isinstance(train_data, (pd.DataFrame, str)):
            raise ValueError(
                "Please provide data training data. This must be a pandas DataFrame or a path to a CSV file."
            )

        if not isinstance(new_data, (pd.DataFrame, str)):
            raise ValueError(
                "Please provide data to be labeled. This must be a pandas DataFrame or a path to a CSV file."
            )

        # Validate training parameters
        if not isinstance(training_params, list) or len(training_params) < 7:
            raise ValueError(
                "training_params must be a list of at least 7 hyperparameter values."
            )

        if not isinstance(split_ratio, float) or not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio must be a float between 0 and 1.")

        # Validate column names
        if not isinstance(text_column, str):
            raise ValueError("text_column must be a string.")
        if not isinstance(label_column, str):
            raise ValueError("label_column must be a string.")

        # Validate columns_to_classify
        if columns_to_classify is not None:
            if not isinstance(columns_to_classify, (str, list)):
                raise ValueError(
                    "columns_to_classify must be a string or a list of strings."
                )
            if isinstance(columns_to_classify, list) and not all(
                isinstance(col, str) for col in columns_to_classify
            ):
                raise ValueError("All entries in columns_to_classify must be strings.")

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
            text_columns=columns_to_classify,
            tokenizer=self.tokenizer,
            label2id=label2id,
            save_path=prediction_save_path,
        )

        return df_annotated


### 4. Visualization and Analysis ####
from educhateval.descriptive_results.display_results import (
    plot_predicted_categories,
    plot_category_bars,
    create_prediction_summary_table,
    plot_previous_turn_distribution,
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
        return plot_predicted_categories(
            df, student_col=student_col, tutor_col=tutor_col, **kwargs
        )

    def plot_category_bars(self, df, student_col=None, tutor_col=None, **kwargs):
        """Wrapper for grouped barplot of predicted categories."""
        return plot_category_bars(
            df, student_col=student_col, tutor_col=tutor_col, **kwargs
        )

    def create_summary_table(self, df, student_col=None, tutor_col=None):
        """Wrapper for generating prediction summary table."""
        return create_prediction_summary_table(
            df, student_col=student_col, tutor_col=tutor_col
        )

    def plot_history_interaction(
        self, df, student_col=None, tutor_col=None, focus_agent="student", **kwargs
    ):
        """Wrapper for barplot showing category transitions from previous turn."""
        return plot_previous_turn_distribution(
            df,
            student_col=student_col,
            tutor_col=tutor_col,
            focus_agent=focus_agent,
            **kwargs,
        )
