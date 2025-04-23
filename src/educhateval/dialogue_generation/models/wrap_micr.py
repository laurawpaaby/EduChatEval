"""
Chat Model Wrapper for MLX supported models
Builds on the codebase by Mina Almasi in https://github.com/INTERACT-LLM/Interact-LLM/
"""

from pathlib import Path
from typing import Optional, List

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler


# packages for structure
from educhateval.dialogue_generation.chat import ChatMessage
from educhateval.dialogue_generation.chat_model_interface import ChatModelInterface


class ChatMLX(ChatModelInterface):
    """
    Model wrapper for loading and using a quantized Hugging Face model through MLX.

    This class implements the ChatModelInterface, allowing it to be used
    interchangeably with other chat models such as ChatHF.
    It is optimized for fast, low-memory inference on Apple Silicon (M1/M2/M3).
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        sampling_params: Optional[dict] = None,
        penalty_params: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.device_map = device_map
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None

        # Optional sampler and logits processor for generation hyperparameters
        self.sampler = make_sampler(**sampling_params) if sampling_params else None
        self.logits_processor = (
            make_logits_processors(**penalty_params) if penalty_params else None
        )

    def load(self) -> None:
        """
        Load the quantized MLX model and tokenizer.
        Only loads if they haven't already been initialized.
        """
        if self.tokenizer is None or self.model is None:
            self.model, self.tokenizer = load(self.model_id)

            if self.device:
                self.model.to(self.device)

    def generate(
        self, chat: List[ChatMessage], max_new_tokens: int = 3000
    ) -> ChatMessage:
        """
        Generate a response based on the given chat history.
        Applies the chat template from the tokenizer, and uses MLX's generate function.

        Returns:
            ChatMessage: the assistant's response
        """
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate response using MLX model
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=False,
            max_tokens=max_new_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processor,
        )

        return ChatMessage(role="assistant", content=response)
