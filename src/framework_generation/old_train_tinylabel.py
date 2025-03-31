
####  -------- Purpose: -------- ####

# 1. Train a classification model on a small manually created data set 
# 2. Get model performance on train and test data
# 3. Save the model and tokenizer 
# 4. Use model on the synthesized dataset to predict each label
# 5. Keep only the ones where the classifier agrees - you now have the final dataset of high quality 

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
# - The final dataset with only the examples where the classifier agrees

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna

def load_dataset(csv_path: str, text_column: str, label_column: str, split_ratio: float):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    unique_labels = df[label_column].nunique()

    # map the labels to integer ids for the model
    label2id = {label: idx for idx, label in enumerate(sorted(df[label_column].unique()))}
    df[label_column] = df[label_column].map(label2id)

    dataset = Dataset.from_pandas(df[[text_column, label_column]])

    # name the label column as 'labels' for the trainer and text column 'text' - but only if not already named that
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if label_column != "labels":
        dataset = dataset.rename_column(label_column, "labels")

    split_dataset = dataset.train_test_split(test_size=split_ratio)
    return DatasetDict({"train": split_dataset["train"], "test": split_dataset["test"]}), unique_labels



# 1.2 Tokenize the dataset
def tokenize_datasets(dataset_dict: DatasetDict, model_name: str, text_column: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add a pad token if it's missing - distilled models don't have them typically
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # to get the max token length for padding
    sample_texts = dataset_dict["train"][text_column]
    token_lengths = [len(tokenizer.encode(text)) for text in sample_texts]
    print("Max token:", max(token_lengths), "Average token:", sum(token_lengths)/len(token_lengths))

    # returning input_ids, attention_mask
    def tokenize_function(examples):
        encoded = tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=max(token_lengths))
        encoded["labels"] = examples["labels"]
        return encoded


    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    return tokenized_datasets, tokenizer

# 2. Train the model
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1) # most confident prediction
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def train_model(model_name: str, dataset_dict: DatasetDict, label_count: int, training_params: list, tuning: bool = False, tuning_params: dict = None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=label_count)

    # using specified params
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
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        compute_metrics=compute_metrics
    )

    # tuning with optuna given grid params
    if tuning:
        print("Starting hyperparameter tuning...")
        def objective(trial):
            for key, values in tuning_params.items():
                setattr(training_args, key, trial.suggest_categorical(key, values))
            trainer.args = training_args
            trainer.train()
            return trainer.evaluate()['eval_loss']

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=len(tuning_params[list(tuning_params.keys())[0]]))
        best_params = study.best_params
        print("Best hyperparameters:", best_params)

    trainer.train()
    return model, trainer

# 3. Save the trained model and tokenizer
def save_model(model, tokenizer, save_path: str):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


# 4. Use model on synthesized data and filter based on agreement
def filter_synthesized_data(synth_path: str, model, tokenizer, label_column: str, save_path: str):
    print("Starting filtering synthesized data based on model predictions...")
    df = pd.read_csv(synth_path, encoding="utf-8", on_bad_lines='skip')

    # Convert label column to integer IDs
    label2id = {label: idx for idx, label in enumerate(sorted(df[label_column].unique()))}
    df["label_id"] = df[label_column].map(label2id)

    dataset = Dataset.from_pandas(df)

    # tokenize text and predict the classes 
    tokenized = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
    trainer = Trainer(model=model)
    predictions = trainer.predict(tokenized)
    preds = predictions.predictions.argmax(-1)
    print(preds)
    print(len(preds))

    # store predicted labels in the dataframe with the original text
    df["predicted"] = preds
    df_filtered = df[df["label_id"] == df["predicted"]]
    print(df_filtered)

    df_filtered.to_csv(save_path, index=False)
    print(f"Filtered data saved to {save_path}")
    return df_filtered


# Main function
def train_and_save_model(csv_path: str, model_name: str, text_column: str, 
                         label_column: str, split_ratio: float, training_params: list, 
                         tuning: bool = False, tuning_params: dict = None, 
                         save_path: str = None, synth_path: str = None, 
                         filtered_save_path: str = None):
    
    # Load and prepare dataset
    dataset_dict, unique_labels = load_dataset(csv_path, text_column, label_column, split_ratio)
    
    # Tokenize
    tokenized_datasets, tokenizer = tokenize_datasets(dataset_dict, model_name, text_column)
    
    # Train model
    model, trainer = train_model(model_name, tokenized_datasets, unique_labels, training_params, tuning, tuning_params)
    final_metrics = trainer.evaluate()

    # Save model and tokenizer
    save_model(model, tokenizer, save_path)
    print("Training completed. Model and tokenizer saved.")

    # If synthetic data is provided, filter it
    if synth_path:
        filtered_data = filter_synthesized_data(synth_path, model, tokenizer, label_column, filtered_save_path)
        return model, tokenizer, final_metrics, filtered_data

    return model, tokenizer, final_metrics





###################################
############## USAGE ##############
###################################

## intent in language
csv_path = "/Users/dklaupaa/Desktop/chat_wrap_package/tiny_labeled_data.csv"
model_name = "distilbert-base-uncased"            # Pretrained model (can be local or Hugging Face model ID)
text_column = "text"
label_column = "category"
split_ratio = 0.2                                  # 80/20 train/test split
save_path = "./my_quality_classifier_model"
synth_path = "/Users/dklaupaa/Desktop/chat_wrap_package/FINAL_synthetic_combined500.csv"
filtered_save_path = "./final_quality_data.csv"

## feedback in language
csv_path_FB = "/Users/dklaupaa/Desktop/chat_wrap_package/tiny_labeled_feedback.csv"  
model_name_FB = "distilbert-base-uncased"           
text_column_FB = "text"
label_column_FB = "overall_label"
split_ratio_FB = 0.2                                  
save_path_FB = "./my_FEEDBACK_quality_classifier_model"
synth_path_FB = "/Users/dklaupaa/Desktop/chat_wrap_package/FINAL_synthetic_feedback1000_wfilters.csv"
filtered_save_path_FB = "./final_FEEDBACK_quality_data.csv"


# Set training parameters: [loss_dummy, loss_name, learning_rate, train_batch, eval_batch, epochs, weight_decay]
training_params = [0.01, 'cross_entropy', 5e-5, 8, 8, 4, 0.01]

# Call the pipeline
model, tokenizer, metrics = train_and_save_model(
    csv_path=csv_path_FB,
    model_name=model_name_FB,
    text_column=text_column_FB,
    label_column=label_column_FB,
    split_ratio=split_ratio_FB,
    training_params=training_params,
    tuning=True,
    tuning_params = {
    'learning_rate': [5e-5, 3e-5, 2e-5],
    'per_device_train_batch_size': [8, 16, 32],
    'num_train_epochs': [3, 5, 10],
    'weight_decay': [0.01, 0.1]}, 
    save_path=save_path_FB,
    synth_path=synth_path_FB,
    filtered_save_path=filtered_save_path_FB
)

# Print final performance
print("Final evaluation metrics:", metrics)