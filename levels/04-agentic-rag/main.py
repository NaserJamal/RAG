"""
Main Runner - AI Agent Demo

Interactive AI agent with tool calling capabilities.

Usage:
    python main.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from interface.conversation import run_conversation
from interface.ui import print_welcome
from core.tool_system import registry

# Import tools to trigger registration
import tools

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / ".env")


def main():
    """Run the AI agent demo."""

    # Display welcome message
    print_welcome(registry)

    # Initialize LLM client (OpenAI-compatible API)
    client = OpenAI(
        base_url=os.getenv('LLM_BASE_URL', 'https://openrouter.ai/api/v1'),
        api_key=os.getenv('LLM_API_KEY')
    )

    # Model configuration
    model = os.getenv('LLM_MODEL', 'qwen/qwen3-coder-30b-a3b-instruct')

    # Start interactive conversation
    run_conversation(client, model)


if __name__ == "__main__":
    main()
