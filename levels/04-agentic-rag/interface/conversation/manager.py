"""
Conversation Manager - Handles interactive chat sessions with the AI agent.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared import Config
from tools.rag.tree_builder import build_document_tree
from core.agent import run_agent_loop
from interface.ui import print_separator, print_user_message, print_session_end, print_error


def load_system_prompt() -> str:
    """
    Load the system prompt from file and inject document tree structure.

    Returns:
        System prompt content with document tree, or empty string if file not found
    """
    prompt_file = Path(__file__).parent / "system_prompt.txt"

    # Load base system prompt
    base_prompt = ""
    if prompt_file.exists():
        base_prompt = prompt_file.read_text().strip()

    # Generate document tree structure
    try:
        documents_path = Config.get_documents_path()
        tree_structure = build_document_tree(documents_path)

        # Inject document tree into system prompt
        tree_section = f"""

## Available Knowledge Base

You have access to the following documents organized by category:

```
{tree_structure}
```

When using search tools, you can optionally filter to a specific file by providing the file_path parameter in the format: 'category/filename.txt' (e.g., 'company-kb/expense-reimbursement.txt').
"""
        return base_prompt + tree_section

    except Exception as e:
        print(f"Warning: Failed to build document tree: {e}")
        return base_prompt


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

    print_separator()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            messages.append({
                "role": "user",
                "content": user_input
            })

            print_user_message(user_input)

            run_agent_loop(client, messages, model)

        except (KeyboardInterrupt, EOFError):
            print_session_end()
            break

        except Exception as e:
            print_error(f"Conversation error: {e}")
            break
