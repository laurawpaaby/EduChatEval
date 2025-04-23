To initiate and wrap a dialogue between a user and LLM-tutor locally, follow this [tutorial](https://github.com/laurawpaaby/EduChatEval/blob/main/tutorials/chat_wrap_instructions.md). 

::: educhateval.chat_ui.ChatWrap


*Example Usage from Terminal*:
```bash
chat-ui \
  --api_url http://127.0.0.1:1234/v1/chat/completions \
  --model llama-3.2-3b-instruct \
  --prompt "You are a helpful tutor guiding a student." \
  --save_dir data/logged_dialogue_data

```