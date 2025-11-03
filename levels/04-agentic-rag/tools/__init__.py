"""
Tools Package - AI Agent Tools

Import this package to automatically register all available tools.
"""

from core.tool_system import registry

# Import all tool modules to trigger registration
from . import sql_tools
from . import weather_tools

__all__ = ['registry']
