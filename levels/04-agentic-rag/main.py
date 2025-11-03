"""
Main Runner - AI Agent Demo

Interactive AI agent with tool calling capabilities.

Usage:
    python main.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from interface.conversation import run_conversation
from interface.ui import display
from core.tool_system import registry

# Import tools to trigger registration
import tools

# Load environment variables
load_dotenv()


def main():
    """Run the AI agent demo."""

    # Display welcome message
    display.print_welcome(registry)

    # Initialize OpenAI client
    client = OpenAI(
        base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
        api_key=os.getenv('OPENROUTER_API_KEY')
    )

    # Model configuration
    model = os.getenv('DEFAULT_MODEL', 'qwen/qwen3-coder-30b-a3b-instruct')

    # Start interactive conversation
    run_conversation(client, model)


if __name__ == "__main__":
    main()
