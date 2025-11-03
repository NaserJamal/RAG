"""Display functions for session lifecycle and system messages."""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

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

    welcome_text = f"""
# AI Agent Demo

This is an interactive AI agent that can use tools to help you.

**Available capabilities:**
{capabilities}
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
