"""
Tools Package - AI Agent Tools

Import this package to automatically register all available tools.
Supports automatic discovery of tools in subdirectories.
"""

import importlib
from pathlib import Path
from core.tool_system import registry


def _discover_and_import_tools():
    """
    Recursively discover and import all Python modules in the tools directory.

    Any module containing @registry.register decorators will be automatically
    registered when imported.
    """
    tools_dir = Path(__file__).parent

    # Find all Python files recursively, excluding __init__.py and __pycache__
    for py_file in tools_dir.rglob("*.py"):
        if py_file.name.startswith("__") or "__pycache__" in str(py_file):
            continue

        # Build module path relative to tools directory
        relative_path = py_file.relative_to(tools_dir)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        module_name = ".".join(module_parts)

        # Import the module (triggers @registry.register decorators)
        try:
            importlib.import_module(f".{module_name}", package=__package__)
        except Exception as e:
            print(f"Warning: Failed to import tool module {module_name}: {e}")


# Auto-discover and register all tools
_discover_and_import_tools()

__all__ = ['registry']
