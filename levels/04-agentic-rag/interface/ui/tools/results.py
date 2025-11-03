"""Display functions for tool results with intelligent formatting."""

import json
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()


def print_tool_result(result: Any):
    """Display a tool result with intelligent formatting."""
    # Handle dictionary results
    if isinstance(result, dict):
        _print_dict_result(result)
    # Handle list results
    elif isinstance(result, list):
        _print_list_result(result)
    # Handle string results (check if it's JSON)
    elif isinstance(result, str):
        try:
            parsed = json.loads(result)
            _print_dict_result(parsed) if isinstance(parsed, dict) else console.print(f"[bold cyan]   ✓ Result:[/bold cyan] {result}")
        except json.JSONDecodeError:
            console.print(f"[bold cyan]   ✓ Result:[/bold cyan] {result}")
    # Handle other types
    else:
        console.print(f"[bold cyan]   ✓ Result:[/bold cyan] {result}")


def print_tool_error(error: str):
    """Display a tool error."""
    console.print(f"[bold red]   ✗ Error:[/bold red] {error}")


def _print_dict_result(result: dict):
    """Pretty print dictionary results with special handling for common patterns."""
    # Check if this is a search result with 'results' key
    if "results" in result and isinstance(result.get("results"), list):
        _print_search_results(result)
    # Check if it contains an error
    elif "error" in result:
        console.print(f"[bold red]   ✗ Error:[/bold red] {result['error']}")
        # Print other fields if present
        for key, value in result.items():
            if key != "error":
                console.print(f"[dim]   {key}:[/dim] {value}")
    # Generic dictionary with few items - print inline
    elif len(result) <= 3:
        console.print(f"[bold cyan]   ✓ Result:[/bold cyan]")
        for key, value in result.items():
            console.print(f"[cyan]     • {key}:[/cyan] {value}")
    # Larger dictionary - use panel with JSON
    else:
        result_json = json.dumps(result, indent=2)
        syntax = Syntax(result_json, "json", theme="monokai", line_numbers=False)
        panel = Panel(
            syntax,
            title="[bold cyan]✓ Result[/bold cyan]",
            border_style="cyan",
            expand=False
        )
        console.print(panel)


def _print_search_results(result: dict):
    """Pretty print search/retrieval results."""
    query = result.get("query", "N/A")
    result_count = result.get("result_count", 0)
    results = result.get("results", [])

    # Print summary
    console.print(f"[bold cyan]   ✓ Found {result_count} result{'s' if result_count != 1 else ''}[/bold cyan] for query: [italic]{query}[/italic]")

    if not results:
        console.print("   [dim]No results to display[/dim]")
        return

    # Print each result as a nice panel
    for idx, item in enumerate(results, 1):
        doc_id = item.get("document_id", "unknown")
        score = item.get("relevance_score", 0.0)
        content = item.get("content", "")

        # Truncate content for display (show first 200 chars)
        display_content = content[:200] + "..." if len(content) > 200 else content

        # Create result text
        result_text = Text()
        result_text.append(f"Document: ", style="bold")
        result_text.append(f"{doc_id}\n", style="cyan")
        result_text.append(f"Relevance: ", style="bold")
        result_text.append(f"{score:.4f}\n\n", style="yellow")
        result_text.append(display_content, style="white")

        panel = Panel(
            result_text,
            title=f"[bold cyan]Result {idx}/{result_count}[/bold cyan]",
            border_style="cyan",
            expand=False
        )
        console.print(panel)


def _print_list_result(result: list):
    """Pretty print list results."""
    console.print(f"[bold cyan]   ✓ Result:[/bold cyan] {len(result)} items")
    for idx, item in enumerate(result[:5], 1):  # Show first 5 items
        console.print(f"   {idx}. {item}")
    if len(result) > 5:
        console.print(f"   [dim]... and {len(result) - 5} more[/dim]")
