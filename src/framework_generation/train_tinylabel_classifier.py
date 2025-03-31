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
import argparse


class QualityFilterPipeline:
    """
    A pipeline class to train a classifier on small labeled data and filter synthetic data based on classifier agreement.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def load_dataset(self, csv_path: str, text_column: str, label_column: str, split_ratio: float):
        """
        Load dataset from CSV and map labels to IDs. Returns DatasetDict and label2id mapping.
        """
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="ISO-8859-1")

        label2id = {label: idx for idx, label in enumerate(sorted(df[label_column].unique()))}
        df[label_column] = df[label_column].map(label2id)

        dataset = Dataset.from_pandas(df[[text_column, label_column]])

        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")
        if label_column != "labels":
            dataset = dataset.rename_column(label_column, "labels")

        split_dataset = dataset.train_test_split(test_size=split_ratio)
        return DatasetDict({"train": split_dataset["train"], "test": split_dataset["test"]}), label2id

    def tokenize_dataset(self, dataset_dict: DatasetDict):
        """
        Tokenize the dataset using max token length from training samples.
        """
        sample_texts = dataset_dict["train"]["text"]
        token_lengths = [len(self.tokenizer.encode(text)) for text in sample_texts]
        max_len = max(token_lengths)
        print("Max token:", max_len, "Average token:", sum(token_lengths)/len(token_lengths))

        def tokenize_function(examples):
            encoded = self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
            encoded["labels"] = examples["labels"]
            return encoded

        return dataset_dict.map(tokenize_function, batched=True)

    def compute_metrics(self, p: EvalPrediction):
        """
        Compute classification performance metrics.
        """
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def train_model(self, tokenized_dataset: DatasetDict, label_count: int, training_params: list, tuning: bool = False, tuning_params: dict = None):
        """
        Train the model with or without Optuna hyperparameter tuning.
        """
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=label_count)

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
            compute_metrics=self.compute_metrics
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

    def save_model(self, model, save_path: str):
        """
        Save the trained model and tokenizer.
        """
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

    def filter_synthesized_data(self, synth_path: str, model, label_column: str, save_path: str):
        """
        Use the trained classifier to filter out low-quality synthetic samples.
        """
        df = pd.read_csv(synth_path, encoding="utf-8", on_bad_lines='skip')
        label2id = {label: idx for idx, label in enumerate(sorted(df[label_column].unique()))}
        df["label_id"] = df[label_column].map(label2id)

        dataset = Dataset.from_pandas(df)
        tokenized = dataset.map(lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
        trainer = Trainer(model=model)
        predictions = trainer.predict(tokenized)
        preds = predictions.predictions.argmax(-1)

        df["predicted"] = preds
        df_filtered = df[df["label_id"] == df["predicted"]]
        df_filtered.to_csv(save_path, index=False)
        print(f"Filtered data saved to {save_path}")
        return df_filtered



# run the whole thingy - should moved when a proper package/entry point is created!!! has been used for testing
def main():
    ### function inputs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--label_column", default="category")
    parser.add_argument("--split_ratio", type=float, default=0.2)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--synth_path", default=None)
    parser.add_argument("--filtered_save_path", default=None)
    parser.add_argument("--tuning", action="store_true")
    args = parser.parse_args()

    training_params = [0.01, 'cross_entropy', 5e-5, 8, 8, 4, 0.01]
    tuning_params = {
        'learning_rate': [5e-5, 3e-5, 2e-5],
        'per_device_train_batch_size': [8, 16, 32],
        'num_train_epochs': [3, 5, 10],
        'weight_decay': [0.01, 0.1]
    }

    # runing the pipeline
    pipeline = QualityFilterPipeline(model_name=args.model_name)
    dataset_dict, label2id = pipeline.load_dataset(args.csv_path, args.text_column, args.label_column, args.split_ratio)
    tokenized = pipeline.tokenize_dataset(dataset_dict)
    model, trainer = pipeline.train_model(tokenized, len(label2id), training_params, args.tuning, tuning_params)
    trainer.evaluate()
    pipeline.save_model(model, args.save_path)

    if args.synth_path:
        pipeline.filter_synthesized_data(args.synth_path, model, args.label_column, args.filtered_save_path)


#if __name__ == "__main__":
#    main()