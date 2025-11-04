"""
Tool Registry - Central system for managing AI agent tools.

This module provides a decorator-based system for registering
tools that the AI agent can use.
"""

from typing import Callable, Dict, Any, List


class ToolRegistry:
    """Registry for managing tools available to the AI agent."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_definitions: List[Dict[str, Any]] = []

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any]
    ):
        """
        Decorator to register a function as a tool.

        Args:
            name: The name of the tool
            description: What the tool does
            parameters: OpenAI-format parameter schema

        Example:
            @registry.register(
                name="add_numbers",
                description="Add two numbers together",
                parameters={
                    "type": "object",
                    "required": ["a", "b"],
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    }
                }
            )
            def add_numbers(a: float, b: float) -> float:
                return a + b
        """
        def decorator(func: Callable) -> Callable:
            # Store the function
            self._tools[name] = func

            # Store the tool definition for the LLM
            self._tool_definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            })

            return func

        return decorator

    def get_tool(self, name: str) -> Callable:
        """Get a registered tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions in OpenAI format."""
        return self._tool_definitions.copy()


# Global registry instance
registry = ToolRegistry()
