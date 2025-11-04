# Adding Custom Tools

## Quick Start

Create a new `.py` file anywhere in the `tools/` directory (including subdirectories). Tools are automatically discovered and registered.

## Basic Template

```python
from core.tool_system import registry

@registry.register(
    name="my_tool",
    description="What your tool does. The LLM uses this to decide when to call it.",
    parameters={
        "type": "object",
        "required": ["param1"],
        "properties": {
            "param1": {
                "type": "string",
                "description": "What this parameter does"
            },
            "param2": {
                "type": "integer",
                "description": "Optional parameter",
                "default": 5
            }
        }
    }
)
def my_tool(param1: str, param2: int = 5):
    """Your tool implementation."""
    return {
        "result": f"Processed {param1} with value {param2}"
    }
```

## That's it!

- Put your file anywhere in `tools/` or subdirectories
- Use `@registry.register` decorator
- The tool auto-registers on startup
- See `tools/rag/` for examples

## Parameter Types

- `"string"` - Text
- `"integer"` - Whole numbers
- `"number"` - Decimals
- `"boolean"` - true/false
- `"array"` - Lists
- `"object"` - Nested structures
