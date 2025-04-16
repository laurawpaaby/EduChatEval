## 📦 Using `educhateval` to Launch the Chat UI from the Terminal

This guide provides step-by-step instructions for researchers or practitioners to install and run the `educhateval` package and launch the Chat UI from the terminal using the provided CLI interface.

---

### Step 1: Install Python 3.12

Make sure you have **Python 3.12** installed on your system. You can download it from:

- https://www.python.org/downloads/release/python-3120/

Or install via [`pyenv`](https://github.com/pyenv/pyenv) if you manage multiple Python versions.

---

###  Step 2: Set Up a Virtual Environment 
*(Not strictly necessary, but recommended)*

```bash
python3.12 -m venv educhateval-env
source educhateval-env/bin/activate  # On Windows: educhateval-env\Scripts\activate
```

---

### Step 3: Install the Package via PyPI

```bash
pip install educhateval
```

This will install all necessary dependencies, including:
- `textual`
- `requests`, `pandas`, `torch`, etc.

---

### Step 4: Start LM Studio

Before running the chat interface, launch [LM Studio](https://lmstudio.ai/):

1. Open the LM Studio app
2. Load a supported instruct model (e.g. `llama-3.2-3b-instruct`)
3. Enable the OpenAI-compatible server:
   - Use port `1234`
   - Leave the default endpoint: `http://127.0.0.1:1234/v1/chat/completions`

---

### Step 5: Launch the Chat UI from Terminal

Once LM Studio is running, launch the chat interface by running:

```bash
chat-ui \
  --api_url http://127.0.0.1:1234/v1/chat/completions \
  --model llama-3.2-3b-instruct \
  --prompt "You are a helpful tutor guiding a student." \
  --save_dir data/logged_dialogue_data
```

This opens a full-screen terminal-based interface. 
You can begin chatting with the model immediately.

---

### ⌨️ Controls

- Type your message in the input field at the bottom
- Press **Enter** to send
- Press **Ctrl+Q** or click the corner button to quit

---

### 💾 Conversation Logging

After quitting, your session is saved automatically:

- `.json` file: structured turn-by-turn log (roles + messages)
- `.csv` file: tabular with `turn`, `student_msg`, `tutor_msg`

Saved to the path specified in `--save_dir`

---

### 📌 Notes

- You must have LM Studio running before launching `chat-ui`
- No Python coding is required to use the tool
- All model settings are passed through the command line
- Be patient, it might take a minute to load the interface 😉

---