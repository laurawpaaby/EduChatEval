####  -------- Purpose: -------- ####

# 1. Train a classification model with specified data
# 2. Get model performance on train and test data
# 3. Save the model and tokenizer for later use on actual interaction data 

####  -------- Inputs: -------- ####
# - Dataset path for training
# - Model name
# - Name of text column and label column
# - Train/test split ratio (split_ratio)
# - Tuning (optional): TRUE/FALSE
#       - If TRUE, grid of hyper parameters 
#       - If FALSE, default hyper parameters
# - Save path for model and tokenizer

####  -------- Outputs: -------- ####
# - Trained model
# - Tokenizer
# - Training and performance metrics (loss, accuracy, etc.)

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna


# 1.1 Load dataset from CSV with fallback encodings and transform it into Hugging Face DatasetDict
def load_dataset(csv_path: str, text_column: str, label_column: str, split_ratio: float):
    """
    Load dataset from CSV with fallback encodings. Then convert pandas DataFrame into Hugging Face DatasetDict.
    Also, calculate unique labels in the label column for the trainer later on.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    unique_labels = df[label_column].nunique()  # Get the number of unique labels for trainer

    # convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[[text_column, label_column]])
    split_dataset = dataset.train_test_split(test_size=split_ratio)
    
    return DatasetDict({"train": split_dataset["train"], "test": split_dataset["test"]}), unique_labels



# 1.2 Tokenize the dataset 
def tokenize_datasets(dataset_dict: DatasetDict, model_name: str, text_column: str):
    """
    Tokenize the dataset using the specified model's tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column], 
                         padding="max_length", 
                         truncation=True)
    
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    return tokenized_datasets, tokenizer



# 2. Train the model with optional hyperparameter tuning and store performance metrics for both training and testing.
def compute_metrics(p: EvalPrediction):
    """
    Compute performance metrics for classification.
    """
    preds = p.predictions.argmax(-1)  # Take the class with the highest probability
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
 
def train_model(model_name: str, dataset_dict: DatasetDict, label_count: int, training_params: list, tuning: bool = False, tuning_params: dict = None):
    """
    Train the model considering either a direct list of parameters 
    or tuning via a grid + store performance metrics.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=label_count)
    
    # Extract training parameters from the list if tuning is off
    if not tuning:
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
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
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs'
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        compute_metrics=compute_metrics  # Pass the compute_metrics function to the Trainer
    )
    
    if tuning:
        def objective(trial):
            # Define hyperparameter search space from the tuning parameters grid
            for key, values in tuning_params.items():
                setattr(training_args, key, trial.suggest_categorical(key, values))
            trainer.args = training_args
            trainer.train()
            return trainer.evaluate()['eval_loss']
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=len(tuning_params[list(tuning_params.keys())[0]]))  # Number of trials based on length of parameter lists
        best_params = study.best_params
        print("Best hyperparameters:", best_params)
    
    trainer.train()
    return model, trainer




# 3. Save the trained model and tokenizer.
def save_model(model, tokenizer, save_path: str):
    """Save the trained model and tokenizer."""
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")



# Main function to process input parameters and manage the training/saving workflow.
def train_and_save_model(csv_path: str, model_name: str, text_column: str, label_column: str, split_ratio: float, training_params: list, tuning: bool = False, tuning_params: dict = None, save_path: str = "./my_finetuned_model"):
    """Full workflow to train, save, and report on a text classification model."""
    dataset_dict, unique_labels = load_dataset(csv_path, text_column, label_column, split_ratio)
    tokenized_datasets, tokenizer = tokenize_datasets(dataset_dict, model_name, text_column)
    model, trainer = train_model(model_name, tokenized_datasets, unique_labels, training_params, tuning, tuning_params)
    
    # Perform final evaluation and get metrics
    final_metrics = trainer.evaluate()

    # Save the model and tokenizer
    save_model(model, tokenizer, save_path)
    print("Training completed. Model and tokenizer saved.")

    # Return model, tokenizer, and final performance metrics
    return model, tokenizer, final_metrics

