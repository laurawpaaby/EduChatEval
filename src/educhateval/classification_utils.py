# ####  -------- Purpose: -------- ####

# 1. A utility script for training a classification models
# 2. It includes functions for loading and preparing datasets, tokenizing, training, and evaluating the model repeatedly used

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna
from typing import Union

import warnings

# Suppress specific UserWarning about `use_mps_device` and general warning about training the model: done because the models are trained and the version is not 5.0 yet
warnings.filterwarnings(
    "ignore",
    message="`use_mps_device` is deprecated and will be removed in version 5.0",
)
warnings.filterwarnings(
    "ignore",
    message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.",
)


# --- Load tokenizer ---
def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


# --- Load and prepare dataset ---
def load_and_prepare_dataset(
    data: Union[str, pd.DataFrame],
    text_column: str,
    label_column: str,
    split_ratio: float,
):
    """
    Load dataset from CSV or DataFrame and map labels to IDs. Returns DatasetDict and label2id mapping.
    """
    if isinstance(data, str):
        try:
            df = pd.read_csv(data, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(data, encoding="ISO-8859-1")
    else:
        df = data.copy()

    label2id = {
        label: idx for idx, label in enumerate(sorted(df[label_column].unique()))
    }
    df[label_column] = df[label_column].map(label2id)

    dataset = Dataset.from_pandas(df[[text_column, label_column]])

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if label_column != "labels":
        dataset = dataset.rename_column(label_column, "labels")

    split_dataset = dataset.train_test_split(test_size=split_ratio)
    return (
        DatasetDict({"train": split_dataset["train"], "test": split_dataset["test"]}),
        label2id,
    )


# --- Tokenize dataset ---
def tokenize_dataset(dataset_dict: DatasetDict, tokenizer):
    """
    Tokenize the dataset using max token length from training samples.
    """
    sample_texts = dataset_dict["train"]["text"]
    token_lengths = [len(tokenizer.encode(text)) for text in sample_texts]
    max_len = max(token_lengths)
    print(
        "Max token:", max_len, "Average token:", sum(token_lengths) / len(token_lengths)
    )

    def tokenize_function(examples):
        encoded = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=max_len
        )
        encoded["labels"] = examples["labels"]
        return encoded

    return dataset_dict.map(tokenize_function, batched=True)


# --- Compute evaluation metrics ---
def compute_metrics(p: EvalPrediction):
    """
    Compute classification performance metrics.
    """
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# --- Train the model ---

# --- Train the model ---
def train_model(
    tokenized_dataset: DatasetDict,
    model_name: str,
    label_count: int,
    training_params: list,
    tuning: bool = False,
    tuning_params: dict = None,
):
    """
    Train the model with or without Optuna hyperparameter tuning.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=label_count
    )

    if not tuning:
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=training_params[3],
            per_device_eval_batch_size=training_params[4],
            num_train_epochs=training_params[5],
            weight_decay=training_params[6],
            logging_dir="./logs",
            learning_rate=training_params[2],
            use_mps_device=True,
        )
    else:
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    if tuning:

        trial_results = []

        def objective(trial):
            # Suggest parameters
            for key, values in tuning_params.items():
                setattr(training_args, key, trial.suggest_categorical(key, values))
            trainer.args = training_args
            trainer.train()

            eval_metrics = trainer.evaluate()

            # Save all relevant metrics for this trial
            trial_results.append({
                "Trial": trial.number,
                "Validation Loss": eval_metrics["eval_loss"],
                "Accuracy": eval_metrics["eval_accuracy"],
                "Precision": eval_metrics["eval_precision"],
                "Recall": eval_metrics["eval_recall"],
                "F1": eval_metrics["eval_f1"],
                **trial.params
            })

            return eval_metrics["eval_loss"]

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=len(tuning_params[list(tuning_params.keys())[0]]),
        )

        # displaying the results for easier review
        study_results_df = pd.DataFrame(trial_results)
        study_results_df = study_results_df.sort_values("Validation Loss")

        print("\n--- Summary of all tuning trials ---\n")
        print(study_results_df.to_string(index=False))

        print("\nBest hyperparameters:", study.best_params)

    #trainer.train()
    return model, trainer
    



# --- Save model and tokenizer ---
def save_model_and_tokenizer(model, tokenizer, save_path: str):
    """
    Save the trained model and tokenizer.
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


# --- Filter synthetic dataset ---
def filter_synthesized_data(
    synth_input: Union[str, pd.DataFrame],
    model,
    tokenizer,
    label_column: str,
    save_path: str = None,
):
    """
    Use the trained classifier to filter out low-quality synthetic samples.
    Accepts a path or a DataFrame. Returns a cleaned pandas DataFrame. Saves to CSV if path is given.
    """
    if isinstance(synth_input, str):
        df = pd.read_csv(synth_input, encoding="utf-8", on_bad_lines="skip")
    else:
        df = synth_input.copy()

    label2id = {
        label: idx for idx, label in enumerate(sorted(df[label_column].unique()))
    }
    df["label_id"] = df[label_column].map(label2id)

    dataset = Dataset.from_pandas(df)
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True),
        batched=True,
    )
    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized)
    preds = predictions.predictions.argmax(-1)

    df["predicted"] = preds
    df_filtered = df[df["label_id"] == df["predicted"]]

    if save_path:
        df_filtered.to_csv(save_path, index=False)
        print(f"Filtered data saved to {save_path}")

    return df_filtered
