"""User interface utilities for interactive search."""


def print_header(title: str, width: int = 80) -> None:
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_step(message: str) -> None:
    """Print a step message."""
    print(f"\n{message}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def get_user_query() -> str:
    """Get search query from user input."""
    print("\n" + "=" * 80)
    print("🔍 Enter your search query (or 'quit' to exit)")
    print("=" * 80)

    query = input("\nQuery: ").strip()

    if query.lower() in ('quit', 'exit', 'q'):
        print("\n👋 Goodbye!")
        return ""

    return query


def display_results(results: list, query: str) -> None:
    """Display search results in a formatted way."""
    print(f"\n✅ Found {len(results)} results for: '{query}'")

    print_header("Top Results")

    for result in results:
        print(f"#{result['rank']} - {result['document_id']} (Score: {result['score']:.4f})")
        print(f"   {result['preview'][:150]}...")
        print()
