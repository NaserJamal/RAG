"""Display functions for user and assistant messages."""

from rich.console import Console

console = Console()


def print_user_message(message: str):
    """Display a user message."""
    console.print(f"\n[bold blue]👤 User:[/bold blue] {message}")


def print_thinking():
    """Display thinking indicator."""
    console.print("\n[dim]💭 Agent thinking...[/dim]")


def print_assistant_message(message: str, stream: bool = False):
    """Display an assistant message."""
    if not stream:
        console.print(f"\n[bold green]🤖 Assistant:[/bold green]")
    console.print(message, end="" if stream else "\n")
