# GeminiAudioAI
Audio AI to practice coding and Data Science in multiple langauges with screen sharing/capture, voice interaction, and prompt versioning.

# Installation

## Poetry (Recommended)

1. **Install Poetry** (if not already installed):

    ```bash
    pip install poetry
    ```

2. **Install Project Dependencies:**

    ```bash
    poetry install
    ```

3. **Activate the Virtual Environment:**

    - With Poetry (requires the shell plugin):

        ```bash
        poetry self add poetry-plugin-shell      # One-time only, if needed
        poetry shell
        ```

    - Or manually:

        - **Windows:**
          ```bash
          .venv\Scripts\activate
          ```
        - **macOS/Linux:**
          ```bash
          source .venv/bin/activate
          ```

4. **[Optional] Export requirements.txt for pip users:**

    ```bash
    poetry self add poetry-plugin-export    # One-time only
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```

---

## Pip/requirements.txt (Alternative)

1. **Export dependencies (if you use Poetry):**

    ```bash
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```

2. **Install dependencies with pip:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate        # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

---

## .env Setup

1. Copy `.env.example` to `.env` and set your required keys (like `GEMINI_API_KEY`, `LANGSMITH_API_KEY`).

---

## Running the Package

```bash
python main.py --mode screen --model_type 1 --prompt_version 2
```

## Screen mode and Model and Prompt version

RUN python main.py --mode screen --model_type 1 --prompt_version 2


## ✅ Enhancements Integrated

| Feature                                         | ✅ Implemented |
|------------------------------------------------|----------------|
| **Monitor 1 selection** (avoids black screen)  | ✅ `sct.monitors[1]` fallback to `[0]` |
| **PNG format for clarity**                     | ✅ `cv2.imencode(".png", ...)` |
| **Skip PIL, use OpenCV directly**              | ✅ No PIL in screen capture path |
| **Sharpening filter via OpenCV**               | ✅ `cv2.filter2D` kernel |
| **RMS-based voice detection**                  | ✅ `_rms()` + threshold (300) |
| **"Are you still there?" prompt**              | ✅ `keep_alive()` logic |
| **Voice response detection (STT)**             | ✅ Detected via transcription in `receive_audio()` |
| **Async-safe audio streaming** with `sounddevice` | ✅ `run_coroutine_threadsafe` using `self.loop` |
| **Play audio via callback stream**             | ✅ `play_audio()` with `OutputStream` |
| **Handles `TokenTracker` if available**        | ✅ Conditional import and usage |
| **Model version switch via `model_type`**      | ✅ `model_map` with validation |
| **Prompt versioning with LangChain/LangSmith** | ✅ Dynamic prompt loading via `LLangC_Prompt_Manager` |

##  Prompt Versioning with LangChain/LangSmith
GeminiAudioAI loads its system prompt dynamically from LangChain/LangSmith, letting you manage and version prompts centrally—without hardcoding them into your codebase.

### How It Works

- <b> Prompt Manager </b>: The app uses a LLangC_Prompt_Manager class to load the correct system prompt version at startup.
- <b> Prompt Version Flag </b>: Use the --prompt_version N argument when launching the app to choose the desired prompt:

    ```bash
    python main.py --mode screen --model_type 1 --prompt_version 1
    python main.py --mode screen --model_type 1 --prompt_version 2
    ```


- <b> Prompt Identifiers </b>: Each prompt version maps to a unique name (and optionally commit hash) in LangSmith.
- <b> How to Update Prompts </b>:
    1. Go to your LangSmith dashboard.

    2. Create or update your prompt in the repo (e.g., geminaudioai_prompt_v2).

    3. Deploy a new commit if you want a specific version.

    4. Use the correct version number (and optionally commit hash) in your app launch or config.

- <b> Tips for Prompt Management </b>:
    1. Always increment the prompt version number when making major changes to the prompt’s structure, tone, or language.

    2. Store sample prompts and commit hashes in your documentation or .env file for easy reference.

    3. If you encounter a 404 or not found error, check that the prompt name and commit hash exist and are public in your LangSmith account.

- <b> Troubleshooting </b>:
    1. 404/400 Errors: Double-check the prompt name, commit hash, and that your LangSmith API key is correct and has access.

    2. Prompt Not Updating: Restart the app, or force reload the prompt by clearing any local caches or .env references.