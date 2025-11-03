"""Display functions for tool calls."""

import json
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


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
