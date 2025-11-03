"""
Tool Executor - Handles execution of tool calls.

This module is responsible for:
1. Converting LLM arguments to proper types
2. Executing tool functions
3. Handling execution errors
4. Formatting results
"""

import json
from typing import Dict, Any

from core.tool_system import registry
from interface.ui import display


def convert_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert string numbers to appropriate numeric types.

    The LLM sometimes returns numbers as strings. This function
    converts them to float/int for proper tool execution.

    Args:
        arguments: Raw arguments from the LLM

    Returns:
        Converted arguments with proper types
    """
    converted = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            # Try to convert numeric strings to float
            if value.replace('.', '', 1).replace('-', '', 1).isdigit():
                converted[key] = float(value)
            else:
                converted[key] = value
        else:
            converted[key] = value
    return converted


def execute_tool_call(tool_call) -> str:
    """
    Execute a single tool call and return the result.

    Args:
        tool_call: The tool call object from OpenAI

    Returns:
        String representation of the result or error message
    """
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    # Display the tool call
    display.print_tool_call(function_name, arguments)

    try:
        # Get the tool function
        tool_function = registry.get_tool(function_name)

        if not tool_function:
            error_msg = f"Tool '{function_name}' not found"
            display.print_tool_error(error_msg)
            return error_msg

        # Convert arguments and execute
        converted_args = convert_arguments(arguments)
        result = tool_function(**converted_args)

        # Display result
        display.print_tool_result(result)
        return str(result)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        display.print_tool_error(error_msg)
        return error_msg
