"""
Display - Beautiful terminal output for the AI agent.

This module uses the 'rich' library for enhanced terminal output.
Install with: pip install rich
"""

import json
from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

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


def print_tool_call(function_name: str, arguments: Dict[str, Any]):
    """Display a tool call in a beautiful panel."""
    args_json = json.dumps(arguments, indent=2)
    syntax = Syntax(args_json, "json", theme="monokai", line_numbers=False)

    panel = Panel(
        syntax,
        title=f"[bold yellow]🔧 Tool Call: {function_name}[/bold yellow]",
        border_style="yellow",
        expand=False
    )
    console.print(panel)


def print_tool_result(result: Any):
    """Display a tool result."""
    console.print(f"[bold cyan]   ✓ Result:[/bold cyan] {result}")


def print_tool_error(error: str):
    """Display a tool error."""
    console.print(f"[bold red]   ✗ Error:[/bold red] {error}")


def print_error(error: str):
    """Display a general error."""
    panel = Panel(
        error,
        title="[bold red]❌ Error[/bold red]",
        border_style="red",
        expand=False
    )
    console.print(panel)


def print_separator():
    """Print a visual separator."""
    console.print("\n" + "─" * 60 + "\n")


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


def print_session_end():
    """Display session end message."""
    console.print("\n[dim]Session ended.[/dim]\n")
