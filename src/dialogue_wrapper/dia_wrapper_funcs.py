### Purpose:
# Provides a class to manage dialogue between a user and a local LLM (via LM Studio API).
# Logs each user-model interaction and returns a DataFrame. Optionally writes it to CSV.

### Input (for DialogueLogger):
# - api_url: URL to the language model's chat endpoint.
# - model_name: String identifying the model (e.g., llama-3.2-3b-instruct:2).
# - temperature: Sampling temperature.
# - context: System/system prompt message defining assistant behavior.

### Method Input (start_dialogue):
# - participant_id: Unique identifier for the user/session.
# - output_path: (Optional) Path to save the CSV log file.

### Output:
# - Returns: pandas.DataFrame with columns [turn, user_msg, tutor_msg].
# - Also saves CSV if output_path is specified.

import requests
import json
import os
import pandas as pd


class DialogueLogger:
    def __init__(self, api_url: str, model_name: str, temperature: float, context: str):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature
        self.context = context
        self.headers = {"Content-Type": "application/json"}
        

    def start_dialogue(self, participant_id: str, output_path: str = None) -> pd.DataFrame:
        messages = []
        logs = []
        turn = 1

        print("Type 'quit' to exit and save the conversation.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break

            messages.append({"role": "user", "content": user_input})
            payload = {
                "model": self.model_name,
                "messages": [{"role": "system", "content": self.context}] + messages,
                "temperature": self.temperature,
                #"max_tokens": 250,
                "stream": False
            }

            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))

            ### append the user input to the logs
            if response.status_code == 200:
                model_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                messages.append({"role": "assistant", "content": model_response})
                logs.append({
                    "turn": turn,
                    "user_msg": user_input,
                    "tutor_msg": model_response
                })
                print(f"Assistant: {model_response}")
                turn += 1
            else:
                print(f"Error: {response.status_code} - {response.text}")
                break

        df = pd.DataFrame(logs)

        ## write as csv if path provided 
        if output_path:
            filename = os.path.join(output_path, f"conversation_{participant_id}.csv")
            df.to_csv(filename, index=False)
            print(f"Conversation saved to {filename}")

        return df
