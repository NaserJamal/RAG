"""
Conversation Manager - Handles interactive chat sessions with the AI agent.
"""

from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

from core.agent import run_agent_loop
from interface.ui import display


def load_system_prompt() -> str:
    """
    Load the system prompt from file.

    Returns:
        System prompt content, or empty string if file not found
    """
    prompt_file = Path(__file__).parent / "system_prompt.txt"

    if prompt_file.exists():
        return prompt_file.read_text().strip()

    return ""


def run_conversation(client: OpenAI, model: str) -> None:
    """
    Run an interactive conversation with the AI agent.

    Args:
        client: OpenAI client instance
        model: Model name to use
    """
    messages: List[Dict[str, Any]] = []

    # Add system prompt if available
    system_prompt = load_system_prompt()
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    display.print_separator()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            messages.append({
                "role": "user",
                "content": user_input
            })

            display.print_user_message(user_input)

            run_agent_loop(client, messages, model)

        except (KeyboardInterrupt, EOFError):
            display.print_session_end()
            break

        except Exception as e:
            display.print_error(f"Conversation error: {e}")
            break
