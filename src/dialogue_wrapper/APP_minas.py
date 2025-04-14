"""
Initial inspiration:https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/#were-in-the-pipe-five-by-five
and Mina Almasi's application in: https://github.com/INTERACT-LLM/Interact-LLM/blob/main/src/interact_llm/app.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, Markdown
from transformers.utils.logging import disable_progress_bar

### FROM OWN FILES - SHOULD BE UPDATED
from .dialogue_generation.chat import ChatHistory, ChatMessage
from .dialogue_generation.txt_llm_inputs.prompt_wrapper import load_prompt_by_id
from .dialogue_generation.models.wrap_huggingface import ChatHF
from .dialogue_generation.models.wrap_micr import ChatMLX

disable_progress_bar()

DEFAULT_PROMPT_VERSION = 3.0


def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--prompt_id", help="id of prompt in toml", type=str, default="A1"
    )
    parser.add_argument(
        "--prompt_version",
        help="version of prompt toml file",
        type=float,
        default=DEFAULT_PROMPT_VERSION,
    )

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args


class QuitScreen(ModalScreen[bool]):
    """
    Screen with a dialog to quit !!
    From: https://textual.textualize.io/guide/screens/#__tabbed_4_4
    """

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Are you sure you want to quit?", id="question"),
            Button("Quit", variant="error", id="quit"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.dismiss(True)
        else:
            self.dismiss(False)


# classes for formatting
class UserMessage(Markdown):
    pass


class Response(Markdown):
    BORDER_TITLE = "Interact-LLM"


class ChatApp(App):
    """
    Texttual app for chatting with llm
    """

    AUTO_FOCUS = "INPUT"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [("q", "request_quit", "Quit")]
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

    def __init__(
        self,
        model: ChatHF | ChatMLX,
        chat_history: Optional[ChatHistory] = None,
        chat_messages_dir: Optional[Path] = None,
    ):
        """
        Initializes the terminal app with a loaded ChatHF or ChatMLX model. The application will not start if the model is not loaded.

        Args:
            model: The loaded language model wrapped in either ChatHF or ChatMLX.
            chat_history: An optional chat history to initialize the application with, e.g., to include a system prompt.
            chat_messages_dir: The directory to save chat messages. If None, chat messages will not be saved.
        """

        super().__init__()
        self.model = model
        self.chat_history = (
            ChatHistory(messages=[]) if chat_history is None else chat_history
        )
        self.chat_messages_dir = chat_messages_dir

        # run prelim checks
        self._check_model_is_loaded()

        if self.chat_messages_dir is not None:
            self._ensure_chat_dir_exists()

    def _check_model_is_loaded(self):
        if self.model.model is None:
            self.exit(
                message="[ERROR:] Chat model is not loaded, ensure this is done before launching app."
            )

    # wrangling chat messages
    def _ensure_chat_dir_exists(self):
        self.chat_messages_dir.mkdir(parents=True, exist_ok=True)

    def update_chat_history(self, chat_message: ChatMessage) -> None:
        """Update chat history with a single new message."""
        self.chat_history.messages.append(chat_message)

    # app buttons
    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-view"):
            yield Response("¿Hola quieres practicar conmigo?")
        yield Input(placeholder="Escribe tu mensaje aquí")
        yield Footer()

    def action_request_quit(self) -> None:
        """Action to display the quit dialog."""

        def check_quit(quit: bool | None) -> None:
            """Called when QuitScreen is dismissed."""
            if quit:
                if self.chat_messages_dir is not None:
                    chat_json = json.dumps(
                        [msg.dict() for msg in self.chat_history.messages],
                        indent=3,
                        ensure_ascii=False,
                    )
                    save_file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
                    with open(
                        self.chat_messages_dir / f"{save_file_name}.json", "w"
                    ) as outfile:
                        outfile.write(chat_json)
                self.exit()

        self.push_screen(QuitScreen(), check_quit)

    ## chatting functionality
    @on(Input.Submitted)
    async def on_input(self, user_message: Input.Submitted) -> None:
        chat_view = self.query_one("#chat-view")
        user_message.input.clear()
        await chat_view.mount(UserMessage(user_message.value))
        await chat_view.mount(response := Response())
        response.anchor()

        self.get_model_response(user_message.value, response)

    @work(thread=True)
    def get_model_response(self, user_message: str, response: Response) -> None:
        """
        Displays model response to user message, updating chat history
        """
        self.update_chat_history(ChatMessage(role="user", content=user_message))

        model_response = self.model.generate(self.chat_history)

        # replace weird <|im_end|>
        model_response.content = model_response.content.replace("<|im_end|>", "")

        # display in APP
        response_content = ""
        for chunk in model_response.content:
            response_content += chunk  # add words in a "stream-like" way
            self.call_from_thread(response.update, response_content)

        # update history again with model response
        self.update_chat_history(model_response)


def main():
    # init cli args
    args = input_parse()

    # load prompt
    prompt_version = args.prompt_version
    prompt_id = args.prompt_id
    prompt_file = (
        Path(__file__).parents[2]
        / "configs"
        / "prompts"
        / f"v{str(prompt_version)}.toml"
    )

    print(
        f"[INFO]: Formatting prompts using toml file version {prompt_version} and prompt id {prompt_id}"
    )

    system_prompt = load_prompt_by_id(
        toml_path=prompt_file, prompt_id=prompt_id, system_prompt=True
    )

    # format initial chat msg w. system prompt
    chat_history = ChatHistory(
        messages=[ChatMessage(role=system_prompt.role, content=system_prompt.content)]
    )

    # define sampler params
    sampling_params = {"temp": 0.8, "top_p": 0.95, "min_p": 0.95, "top_k": 40}
    penality_params = {"repetition_penalty": 1.1}

    # load model with MLX if possible, default to HF instead
    try:
        model_id = "mlx-community/Qwen2.5-7B-Instruct-1M-4bit"
        model = ChatMLX(
            model_id=model_id,
            sampling_params=sampling_params,
            penalty_params=penality_params,
        )
        print(f"[INFO]: Loading model {model_id} ... please wait")
        model.load()
    except Exception as e:
        print(f"[INFO:] Failed to run using MLX. Defaulting to HuggingFace. Error: {e}")
        model_id = "BSC-LT/salamandra-2b-instruct"
        cache_dir = Path(__file__).parents[3] / "models"
        model = ChatHF(model_id=model_id, cache_dir=cache_dir)
        print(f"[INFO]: Loading model {model_id} ... please wait")
        model.load()

    # define save dir
    save_dir = (
        Path(__file__).parents[3]
        / "data"
        / model_id.replace("/", "--")
        / f"v{str(prompt_version)}"
        / prompt_id
    )

    # open tui app -> pass loaded model
    app = ChatApp(
        model=model,
        chat_history=chat_history,
        chat_messages_dir=save_dir,
    )
    app.run()


if __name__ == "__main__":
    main()
