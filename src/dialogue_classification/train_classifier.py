# THESE ARE SOME OF THE SAME FUNCTIONS AS IN THE TRAING_TINYLABEL_CLASSIFIER.PY - MAYBE MAKE A BASE SCRIPT FOR THEM WITH COMMON FUNCTIONS/ABSTRACT METHODS
####  -------- Purpose: -------- ####

# 1. Train a classification model with specified data
# 2. Get model performance on train and test data
# 3. Optionally store the model 
# 4. Use model on the interaction dataset to predict each label

####  -------- Inputs: -------- ####
# - Dataset path or df for training
# - Model name
# - Name of text column and label column
# - Train/test split ratio (split_ratio)
# - Tuning (optional): TRUE/FALSE
#       - If TRUE, grid of hyper parameters 
#       - If FALSE, default hyper parameters
# - Save path for model and tokenizer (optional)

####  -------- Outputs: -------- ####
# - Trained model
# - Tokenizer
# - Training and performance metrics (loss, accuracy, etc.)
# - The final dataset with predicted annotations 

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna
from typing import Union, Optional


########## load basic funcs FROM CLASSIFICATION_UTILS INSTEAD LAURA !!!!! ##########
#from classification_utils import (load_tokenizer, load_and_prepare_dataset, tokenize_dataset, 
#    compute_metrics, train_model, save_model_and_tokenizer)
# or maybe first relevant in the entry point of the pipeline not sure, look it up


# --- Load tokenizer ---
def load_tokenizer(model_name: str):
    """
    Load and return a tokenizer based on the specified pre-trained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

# --- Load and prepare dataset ---
def load_and_prepare_dataset(data: Union[str, pd.DataFrame], text_column: str, label_column: str, split_ratio: float):
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

    # label encoding
    label2id = {label: idx for idx, label in enumerate(sorted(df[label_column].unique()))}
    df[label_column] = df[label_column].map(label2id)

    dataset = Dataset.from_pandas(df[[text_column, label_column]])

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if label_column != "labels":
        dataset = dataset.rename_column(label_column, "labels")

    split_dataset = dataset.train_test_split(test_size=split_ratio)
    return DatasetDict({"train": split_dataset["train"], "test": split_dataset["test"]}), label2id


# --- Tokenize dataset ---
def tokenize_dataset(dataset_dict: DatasetDict, tokenizer):
    """
    Tokenize the dataset using max token length from training samples.
    """
    sample_texts = dataset_dict["train"]["text"]
    token_lengths = [len(tokenizer.encode(text)) for text in sample_texts]
    max_len = max(token_lengths)
    print("Max token:", max_len, "Average token:", sum(token_lengths)/len(token_lengths))

    def tokenize_function(examples):
        encoded = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


# --- Train the model ---
def train_model(tokenized_dataset: DatasetDict, model_name: str, label_count: int, training_params: list, tuning: bool = False, tuning_params: dict = None):
    """
    Train the model with or without Optuna hyperparameter tuning.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=label_count)

    if not tuning:
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=training_params[3],
            per_device_eval_batch_size=training_params[4],
            num_train_epochs=training_params[5],
            weight_decay=training_params[6],
            logging_dir='./logs',
            learning_rate=training_params[2]
        )
    else:
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs'
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    if tuning:
        def objective(trial):
            for key, values in tuning_params.items():
                setattr(training_args, key, trial.suggest_categorical(key, values))
            trainer.args = training_args
            trainer.train()
            return trainer.evaluate()['eval_loss']

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=len(tuning_params[list(tuning_params.keys())[0]]))
        print("Best hyperparameters:", study.best_params)

    trainer.train()
    return model, trainer


# --- Save model and tokenizer ---
def save_model_and_tokenizer(model, tokenizer, save_path: Optional[str]):
    """
    Optionally save the trained model and tokenizer.
    """
    if save_path:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")


# --- Predict and filter dataset ---
def predict_annotated_dataset(new_data: Union[str, pd.DataFrame], model, text_column, tokenizer, label2id, save_path: Optional[str] = None):
    """
    Predict the labels on a new dataset without labels and filter the dataset to only include confident predictions.
    """
    if isinstance(new_data, str):
        df = pd.read_csv(new_data)
    else:
        df = new_data

    # Prepare and tokenize new data
    tokenized = tokenizer(df[text_column].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    predictions = model(**tokenized)
    predicted_labels = predictions.logits.argmax(-1).numpy()
    predicted_label_names = [list(label2id.keys())[label] for label in predicted_labels]

    # Append predicted labels to the DataFrame
    df['predicted_labels'] = predicted_label_names

    # Optionally save the DataFrame
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Predicted data saved to {save_path}")

    return df