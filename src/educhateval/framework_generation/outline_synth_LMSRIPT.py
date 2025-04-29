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
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError
import importlib.util
import sys


# --- Pydantic schema ---
class ConversationAnnotation(BaseModel):
    text: str
    category: str


def generate_from_prompt(
    prompt: str,
    category: str,
    model_name: str,
    api_url: str,
    num_samples: int = 1000,
    temperature: float = 0.85,
    top_p: float = 0.90,
    max_tokens: int = 40,
) -> List[Dict]:
    """
    Generate synthetic examples from a prompt in a constraint scheme using a local LLM.
    """
    results = []
    for _ in range(num_samples):
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": ["<|im_end|>"],
        }
        try:
            response = requests.post(api_url, json=payload)
            raw_text = response.json()["choices"][0]["text"].strip()
            validated = ConversationAnnotation(text=raw_text, category=category)
            results.append(validated.model_dump())
        except (KeyError, ValidationError) as e:
            print(f"Skipping invalid output for category '{category}':", e)
    return results


def synthesize_dataset(
    prompt_dict: Optional[Dict[str, str]] = None,
    prompt_path: Optional[str] = None,
    model_name: str = "llama-3.2-3b-instruct",
    num_samples: int = 500,
    api_url: str = "http://localhost:1234/v1/completions",
    json_out: str = None,
    csv_out: str = None,
    temperature: float = 0.85,
    top_p: float = 0.90,
    max_tokens: int = 40,
) -> pd.DataFrame:
    """
    Generate synthetic data for each prompt-category pair.
    Returns: cleaned pd.DataFrame of generated samples.
    Optionally saves the result to disk if json_out or csv_out is provided.
    """

    # Load prompt dict from path if not provided directly
    if prompt_dict is None:
        if not prompt_path:
            raise ValueError("Either prompt_dict or prompt_path must be provided.")
        prompt_dict = load_prompt_dict_from_py(prompt_path)

    all_data = []
    for category, prompt in prompt_dict.items():
        print(f"Generating for category: {category}")
        examples = generate_from_prompt(
            prompt, category, model_name, api_url, num_samples, temperature, top_p, max_tokens
        )
        all_data.extend(examples)

    df = pd.DataFrame(all_data).drop_duplicates()

    # Clean up unwanted tokens
    df["text"] = df["text"].str.replace("<\\|im_start\\|>", "", regex=True)
    df["text"] = df["text"].str.replace("<\\|im_end\\|>", "", regex=True)

    # RULE: remove rows that contain 'feedback' if Feedback is a category
    if "Feedback" in prompt_dict:
        df = df[~df["text"].str.contains("feedback", case=False)]

    print(f"Number of duplicates removed: {len(all_data) - len(df)}")

    # Optional saving
    if json_out:
        with open(json_out, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=4)
        print(f"\u2192 Saved JSON: {json_out}")

    if csv_out:
        df.to_csv(csv_out, index=False)
        print(f"\u2192 Saved CSV: {csv_out}")

    return df


def load_prompt_dict_from_py(path: str) -> Dict[str, str]:
    """
    Dynamically loads a dictionary named `prompt_dict` from a Python file.
    """
    spec = importlib.util.spec_from_file_location("prompt_module", path)
    prompt_module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_module"] = prompt_module
    spec.loader.exec_module(prompt_module)
    return prompt_module.prompt_dict
