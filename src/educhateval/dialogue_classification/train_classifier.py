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
from typing import Union, Optional, List
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings(
    "ignore",
    message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.",
)

########## all basic functions can be found in classification_utils.py ##########


# --- Predict and annotate dataset ---
def predict_annotated_dataset(
    new_data: Union[str, pd.DataFrame],
    model,
    text_columns: Union[str, List[str]],
    tokenizer,
    label2id,
    save_path: Optional[str] = None,
):
    """
    Predict the labels on a new dataset using one or more text columns.
    Adds both the predicted label and its confidence score.
    Compatible with Apple M1/M2 (MPS). Optionally saves output as CSV.
    """
    # Load data from path or use provided DataFrame
    if isinstance(new_data, str):
        df = pd.read_csv(new_data)
    else:
        df = new_data

    # Ensure text_columns is a list
    if isinstance(text_columns, str):
        text_columns = [text_columns]

    # Move model to appropriate device (MPS if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Create a copy of the original DataFrame to append predictions
    df_predictions = df.copy()

    for column in text_columns:
        # Tokenize and move to device
        tokenized = tokenizer(
            df[column].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        # Run model in inference mode
        with torch.no_grad():
            logits = model(**tokenized).logits
            probs = F.softmax(logits, dim=-1)
            predicted_labels = probs.argmax(dim=-1).cpu().numpy()
            predicted_confidences = probs.max(dim=-1).values.cpu().numpy()

        # Map label indices back to names
        predicted_label_names = [list(label2id.keys())[i] for i in predicted_labels]

        # Append to DataFrame
        df_predictions[f"predicted_labels_{column}"] = predicted_label_names
        df_predictions[f"predicted_confidence_{column}"] = predicted_confidences

    # Save predictions to CSV if requested
    if save_path:
        df_predictions.to_csv(save_path, index=False)
        print(f"Predicted data saved to {save_path}")

    return df_predictions
