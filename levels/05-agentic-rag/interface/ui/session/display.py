"""Display functions for session lifecycle and system messages."""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from shared import Config
from tools.rag.utils import build_document_tree

console = Console()


def print_welcome(registry=None):
    """Display welcome message with dynamic tool capabilities."""
    capabilities = ""
    if registry:
        tools = registry.get_tool_definitions()
        capabilities = "\n".join(
            f"- {tool['function']['description']}"
            for tool in tools
        )

    # Generate document tree structure
    tree_structure = ""
    try:
        documents_path = Config.get_documents_path()
        tree = build_document_tree(documents_path)
        tree_structure = f"""

**Knowledge Base Structure:**
```
{tree}
```
"""
    except Exception:
        # Silently skip if tree cannot be built
        pass

    welcome_text = f"""
# AI Agent Demo

This is an interactive AI agent that can use tools to help you.

**Available capabilities:**
{capabilities}
{tree_structure}
    """

    md = Markdown(welcome_text)
    panel = Panel(
        md,
        title="[bold magenta]Welcome[/bold magenta]",
        border_style="magenta",
        expand=False
    )
    console.print(panel)


def print_separator():
    """Print a visual separator."""
    console.print("\n" + "─" * 60 + "\n")


def print_session_end():
    """Display session end message."""
    console.print("\n[dim]Session ended.[/dim]\n")


def print_error(error: str):
    """Display a general error."""
    panel = Panel(
        error,
        title="[bold red]❌ Error[/bold red]",
        border_style="red",
        expand=False
    )
    console.print(panel)
