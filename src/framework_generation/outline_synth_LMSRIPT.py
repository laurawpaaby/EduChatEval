####  -------- Purpose: -------- ####

# Synthesize a labeled dataset of a theoretical framework for later model training 
# Method: constraint synthesis using ouline: https://github.com/dottxt-ai/outlines?tab=readme-ov-file#type-constraint 

# 1. Load relevant generative model from lm studio
# 2. Synthesize text in constraint way where only defined categories are generated

####  -------- Inputs: -------- ####
# - Categories to generate
# - Generative Model Name
# - Number of examples to generate
# - Text generation parameters (optional)
# - Save path for dataset


####  -------- Outputs: -------- ####
# - Labeled dataset for training in the format of a CSV file or json

import json
import argparse
import requests
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel, ValidationError
import importlib.util
import sys
import os


# --- Pydantic schema ---
class ConversationAnnotation(BaseModel):
    text: str
    category: str


class SyntheticDataGenerator:
    def __init__(self, model_name: str = "llama-3.2-3b-instruct", api_url: str = "http://localhost:1234/v1/completions"):
        self.model_name = model_name
        self.api_url = api_url

    def generate_from_prompt(self, prompt: str, category: str, num_samples: int = 1000) -> List[Dict]:
        results = []
        for _ in range(num_samples):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.85,
                "top_p": 0.90,
                "max_tokens": 40,
                "stop": ["<|im_end|>"]
            }
            try:
                response = requests.post(self.api_url, json=payload)
                raw_text = response.json()["choices"][0]["text"].strip()
                validated = ConversationAnnotation(text=raw_text, category=category)
                results.append(validated.model_dump())
            except (KeyError, ValidationError) as e:
                print(f"Skipping invalid output for category '{category}':", e)
        return results

    def synthesize_dataset(self, prompt_dict: Dict[str, str], save_json_path: str, save_csv_path: str, num_samples: int = 1000):
        """
        Generate synthetic data for each prompt-category pair and save to disk.
        """
        all_data = []
        for category, prompt in prompt_dict.items():
            print(f"Generating for category: {category}")
            examples = self.generate_from_prompt(prompt, category, num_samples)
            all_data.extend(examples)

        with open(save_json_path, "w") as f:
            json.dump(all_data, f, indent=4)

        # Convert to DataFrame and clean
        df = pd.DataFrame(all_data)

        # Remove duplicates based on text input and category
        df = df.drop_duplicates()

        # Clean up weird tokens that may sneak in
        df["text"] = df["text"].str.replace("<\\|im_start\\|>", "", regex=True)
        df["text"] = df["text"].str.replace("<\\|im_end\\|>", "", regex=True)

        # RULE FOR NOW AT LEAST filter out rows that contain specific keywords like 'feedback' 
        # -> cause it indicates that the model meta explains it self, which we don't want 
        if "Feedback" in prompt_dict:
            df = df[~df["text"].str.contains("feedback", case=False)]

        # print and save 
        print(f"Number of duplicates removed: {len(all_data) - len(df)}")
        df.to_csv(save_csv_path, index=False)

        print(f"Saved {len(df)} total samples to:")
        print(f"\u2192 JSON: {save_json_path}")
        print(f"\u2192 CSV: {save_csv_path}")


# function to load the prompt dictionary from a python file 
def load_prompt_dict_from_py(path):
    spec = importlib.util.spec_from_file_location("prompt_module", path)
    prompt_module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_module"] = prompt_module
    spec.loader.exec_module(prompt_module)
    return prompt_module.prompt_dict

# run the whole thingy - should moved when a proper package/entry point is created
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic labeled text data using a local LM Studio model.")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to JSON file containing prompts.")
    parser.add_argument("--json_out", type=str, required=True, help="Path to save the JSON output.")
    parser.add_argument("--csv_out", type=str, required=True, help="Path to save the CSV output.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate per category.")
    parser.add_argument("--model", type=str, default="llama-3.2-3b-instruct", help="Model name used for generation.")

    args = parser.parse_args()

    prompt_dict = load_prompt_dict_from_py(args.prompt_path)

    # run time 
    generator = SyntheticDataGenerator(model_name=args.model)
    generator.synthesize_dataset(prompt_dict, args.json_out, args.csv_out, args.samples)
