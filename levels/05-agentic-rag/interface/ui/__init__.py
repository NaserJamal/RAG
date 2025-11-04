"""
UI Module - Clean terminal output for the AI agent.

This module provides organized display components grouped by domain:
- messages: User/assistant message display
- tools: Tool call and result formatting
- session: Welcome screens, separators, lifecycle messages
"""

# Import all display functions from domain modules
from .messages import (
    print_user_message,
    print_assistant_message,
    print_thinking,
)

from .tools import (
    print_tool_call,
    print_tool_result,
    print_tool_error,
)

from .session import (
    print_welcome,
    print_separator,
    print_session_end,
    print_error,
)

__all__ = [
    # Messages
    "print_user_message",
    "print_assistant_message",
    "print_thinking",
    # Tools
    "print_tool_call",
    "print_tool_result",
    "print_tool_error",
    # Session
    "print_welcome",
    "print_separator",
    "print_session_end",
    "print_error",
]
