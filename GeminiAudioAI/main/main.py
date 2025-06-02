# Import audio_loop relative to this directory
import argparse
import asyncio
import os
import sys

from audio_loop import AudioLoop
from dotenv import load_dotenv
from prompt_manager import LLangC_Prompt_Manager

# Try importing token_tracker from project root
try:
    from token_tracker import TokenTracker
except ImportError:
    TokenTracker = None
    print("Warning: token_tracker module not found. Token tracking will be disabled.")


load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="screen",
        choices=["camera", "screen", "none"],
        help="Pixels to use",
    )
    parser.add_argument(
        "--model_type",
        type=int,
        default=1,
        choices=[1, 2],
        help="Type of model to use (1 for thinking 2 for non-thinking)",
    )
    parser.add_argument(
        "--prompt_version",
        type=int,
        default=2,
        choices=[1, 2],
        help="The version of the prompt that is loaded on llangchain",
    )
    args = parser.parse_args()

    prompt_manager = LLangC_Prompt_Manager(version=args.prompt_version)
    prompt_manager.load_prompt_name()
    prompt_manager.get_llang_chain_access()
    prompt_text = prompt_manager.prompt_template  # Get the prompt string

    main = AudioLoop(video_mode=args.mode)
    main.model_type = args.model_type
    main.prompt_version = args.prompt_version
    main.prompt = prompt_text

    asyncio.run(main.run())
