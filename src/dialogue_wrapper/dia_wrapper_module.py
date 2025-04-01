####import json
import requests
import argparse
import os
import json 


# THIS IS CURRETNLY A SCRIPT of funcs - consider making it module based to be called with inputs from terminal

def chat_with_model(api_url, model_name, temperature, context, participant_id, output_path):
    headers = {"Content-Type": "application/json"}
    conversation = []
    
    print("Type 'quit' to exit and save the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        conversation.append({"role": "user", "content": user_input})
        
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": context}] + conversation,
            "temperature": temperature,
            "max_tokens": 50,
            "stream": False
        }
        
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            response_data = response.json()
            model_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            conversation.append({"role": "assistant", "content": model_response})
            print(f"Assistant: {model_response}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    # Save the conversation as JSON with participant ID in filename
    output_filename = os.path.join(output_path, f"conversation_{participant_id}.ndjson")
    with open(output_filename, "w") as f:
        json.dump(conversation, f, indent=4)
    
    print(f"Conversation saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interactive chat with a language model.")
    parser.add_argument("--api_url", type=str, required=True, help="API endpoint for the language model")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama-3.2-3b-instruct:2)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature setting for model responses")
    parser.add_argument("--context", type=str, required=True, help="Context description for the model")
    parser.add_argument("--participant_id", type=str, required=True, help="Unique identifier for the participant")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the conversation log")
    
    args = parser.parse_args()
    
    chat_with_model(
        api_url=args.api_url,
        model_name=args.model_name,
        temperature=args.temperature,
        context=args.context,
        participant_id=args.participant_id,
        output_path=args.output_path
    )


# RAN LIKE THIS FROM TERMINAL :
# python chat_src/dynam_llm_stud.py \
  #  --api_url "http://127.0.0.1:1234/v1/chat/completions" \
  #  --model_name "llama-3.2-3b-instruct:2" \
  #  --temperature 0.8 \
  #  --context "You are a helpful assistant." \
  #  --participant_id "12345" \
  #  --output_path "/Users/dklaupaa/Desktop/Thesis_Testing2025/logged_chats_txt"