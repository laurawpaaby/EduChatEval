"""
Terminal Chat UI with LM Studio model via API and Textual frontend.

- Interactive full-screen chat interface
- Saves chat history as JSON and CSV
- Built on your existing ChatMessage + ChatHistory schema

Initial inspiration:https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five
and Mina Almasi's application in: https://github.com/INTERACT-LLM/Interact-LLM/blob/main/src/interact_llm/app.py
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Grid
from textual.widgets import Input, Footer, Markdown, Button, Label
from textual.screen import ModalScreen

# Internal imports 
from dialogue_generation.chat import ChatMessage, ChatHistory
from dialogue_generation.chat_model_interface import ChatModelInterface


# ## LM Studio Model Wrapper ##
class ChatLMStudio(ChatModelInterface):
    def __init__(self, api_url: str, model_name: str, temperature: float = 0.7):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature

    def load(self):
        pass  # No loading needed for LM Studio API :D

    def generate(self, chat: ChatHistory, max_new_tokens: int = 500) -> ChatMessage:
        payload = {
            "model": self.model_name,
            "messages": [msg.dict() for msg in chat.messages],
            "temperature": self.temperature,
            "max_tokens": max_new_tokens,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            raise ValueError(f"LM Studio error: {response.status_code} - {response.text}")

        content = response.json()["choices"][0]["message"]["content"]
        return ChatMessage(role="assistant", content=content)


# ## Save CSV ##
def save_chat_as_csv(messages: list[ChatMessage], output_path: Path):
    rows = []
    turn = 1
    for i in range(0, len(messages) - 1, 2):
        user = messages[i]
        assistant = messages[i + 1]
        if user.role == "user" and assistant.role == "assistant":
            rows.append({
                "turn": turn,
                "student_msg": user.content,
                "tutor_msg": assistant.content
            })
            turn += 1
    pd.DataFrame(rows).to_csv(output_path, index=False)


# ## Quit Dialog ##
class QuitScreen(ModalScreen[bool]):
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Are you sure you want to quit?", id="question"),
            Button("Quit", variant="error", id="quit"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "quit")


# ## Markdown Boxes ##
class UserMessage(Markdown): pass
class Response(Markdown): pass


# ## Main App ##
class ChatApp(App):
    CSS = """
    UserMessage {
        background: $primary 10%;
        color: $text;
        margin: 1;
        margin-right: 8;
        padding: 1 2 0 2;
    }

    Response {
        border: wide $success;
        background: $success 10%;
        color: $text;
        margin: 1;
        margin-left: 8;
        padding: 1 2 0 2;
    }

    QuitScreen {
        align: center middle;
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        width: 60;
        height: 11;
        border: thick $background 80%;
        background: $surface;
    }

    #question {
        column-span: 2;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
    }

    Button {
        width: 100%;
    }
    """

    BINDINGS = [("q", "request_quit", "Quit")]

    def __init__(
        self,
        model: ChatLMStudio,
        chat_history: Optional[ChatHistory] = None,
        chat_messages_dir: Optional[Path] = None
    ):
        super().__init__()
        self.model = model
        self.chat_history = chat_history or ChatHistory(messages=[])
        self.chat_messages_dir = chat_messages_dir
        if self.chat_messages_dir:
            self.chat_messages_dir.mkdir(parents=True, exist_ok=True)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-view"):
            yield Response("Hi, ask me anything or type '^q' to quit.")
        yield Input(placeholder="Write your message here...")
        yield Footer()

    def update_chat(self, chat_message: ChatMessage):
        self.chat_history.messages.append(chat_message)

    async def action_request_quit(self):
        result = await self.push_screen(QuitScreen())
        print(f"[DEBUG] Quit confirmed? {result}")  # This should show True/False BUT NOTHING IS PRINTED...

        if result:
            if self.chat_messages_dir:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                messages = self.chat_history.messages

                # Save JSON
                with open(self.chat_messages_dir / f"{timestamp}.json", "w", encoding="utf-8") as f:
                    json.dump([m.dict() for m in messages], f, indent=2, ensure_ascii=False)


                # Save CSV WHY ISNT THIS WORKING ?!?!??!!?!
                save_chat_as_csv(messages, self.chat_messages_dir / f"{timestamp}.csv")

            self.exit()


        #self.push_screen(QuitScreen(), quit_callback)


    @on(Input.Submitted)
    async def on_input(self, event: Input.Submitted):
        chat_view = self.query_one("#chat-view")
        msg = event.value.strip()
        if not msg:
            return

        event.input.value = ""
        await chat_view.mount(UserMessage(msg))
        await chat_view.mount(resp := Response())
        resp.anchor()

        self.process_response(msg, resp)

    @work(thread=True)
    def process_response(self, msg: str, response_box: Response):
        self.update_chat(ChatMessage(role="user", content=msg))
        response = self.model.generate(self.chat_history)
        response.content = response.content.replace("<|im_end|>", "")

        final_text = ""
        for char in response.content:
            final_text += char
            self.call_from_thread(response_box.update, final_text)

        self.update_chat(response)


# ## Launch function ##
def main():
    # LM Studio config
    api_url = "http://127.0.0.1:1234/v1/chat/completions"
    model_name = "llama-3.2-3b-instruct"
    temperature = 0.7
    system_prompt = "You are a helpful tutor guiding a student. Answer short and concisely."

    # Init model + prompt
    model = ChatLMStudio(api_url, model_name, temperature)
    history = ChatHistory(messages=[ChatMessage(role="system", content=system_prompt)])

    # Save path 
    save_dir = Path("data/logged_dialogue_data")

    # Launch TUI
    app = ChatApp(model=model, chat_history=history, chat_messages_dir=save_dir)
    app.run()


if __name__ == "__main__":
    main()
