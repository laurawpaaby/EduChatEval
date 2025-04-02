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


########## all basic functions can be found in classification_utils.py ##########

import torch

# --- Predict and filter dataset ---
def predict_annotated_dataset(new_data: Union[str, pd.DataFrame], model, text_column, tokenizer, label2id, save_path: Optional[str] = None):
    """
    Predict the labels on a new dataset without labels and return annotated DataFrame.
    Compatible with Apple M1/M2 (MPS). Optionally saves output as CSV.
    """
    # Load data from path or use provided DataFrame
    if isinstance(new_data, str):
        df = pd.read_csv(new_data)
    else:
        df = new_data

    # Move model and input to appropriate device (MPS if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Tokenize text and move tensors to device
    tokenized = tokenizer(df[text_column].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    # Run prediction without tracking gradients
    with torch.no_grad():
        predictions = model(**tokenized)

    predicted_labels = predictions.logits.argmax(-1).cpu().numpy()  # move to CPU to use in pandas
    predicted_label_names = [list(label2id.keys())[label] for label in predicted_labels]

    # Append predicted labels to DataFrame
    df['predicted_labels'] = predicted_label_names

    # Save to CSV if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Predicted data saved to {save_path}")

    return df
