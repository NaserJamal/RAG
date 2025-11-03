# AI Agent with Tool Calling

A clean, educational example of an AI agent that can use tools to solve problems.

## Project Structure

```
tool_calling/
├── main.py                          # Entry point
│
├── core/                            # Core framework
│   ├── agent/                       # Agent orchestration
│   │   ├── loop.py                 # Main agent loop
│   │   └── executor.py             # Tool execution
│   │
│   └── tool_system/                 # Tool infrastructure
│       └── registry.py             # Tool registration system
│
├── tools/                           # Tool implementations
│   ├── sql_tools.py                # SQL query execution
│   └── weather_tools.py            # Weather information
│
├── interface/                       # User interface
│   ├── ui/
│   │   └── display.py              # Terminal output
│   │
│   └── conversation/
│       └── manager.py              # Conversation flow
│
├── .env.example
├── requirements.txt
└── README.md
```

## Features

- **Layered Architecture**: Clean separation between core framework, tools, and interface
- **Interactive Conversations**: Multi-turn chat sessions with conversation history
- **Easy Tool Addition**: Simply decorate a function to register it as a tool
- **Beautiful UI**: Rich terminal output with colors and formatting
- **Streaming**: Real-time response streaming from the LLM
- **Robust Error Handling**: Graceful handling of tool execution errors

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your configuration:
   ```
   OPENROUTER_API_KEY=sk-or-v1-YOUR-ACTUAL-API-KEY
   SQL_TOOL_DATA_BASE_URL=postgresql://username:password@localhost:5432/database
   ```

## Usage

```bash
python main.py
```

Type your requests and have a conversation with the agent. Press Ctrl+C to end the session.

## Adding New Tools

Create a new file in `tools/` and decorate your function:

```python
from core.tool_system import registry

@registry.register(
    name="your_tool_name",
    description="What your tool does",
    parameters={
        "type": "object",
        "required": ["param1"],
        "properties": {
            "param1": {
                "type": "number",
                "description": "Description of parameter"
            }
        }
    }
)
def your_tool_name(param1: float) -> float:
    """Your tool implementation."""
    return param1 * 2
```

Import it in `tools/__init__.py` and it's automatically available to the agent!

## Available Tools

- `execute_sql`: Execute SQL queries against a PostgreSQL database
- `get_weather`: Fetch current weather information for any location

## How It Works

1. **User sends a message** → Agent receives the request
2. **Agent calls LLM** → LLM decides which tools to use
3. **Tools execute** → Functions run and return results
4. **Results sent back** → LLM receives tool outputs
5. **Final response** → LLM formulates answer for user

This cycle repeats until the LLM has all the information it needs!

## Example Session

```
────────────────────────────────────────────────────────────

You: Show me all users from the users table

👤 User: Show me all users from the users table

💭 Agent thinking...

🔧 Tool Call: execute_sql
   Arguments: {
     "query": "SELECT * FROM users LIMIT 10"
   }
   ✓ Result: {
     "rows": [
       {"id": 1, "name": "Alice", "email": "alice@example.com"},
       {"id": 2, "name": "Bob", "email": "bob@example.com"}
     ],
     "row_count": 2
   }

💭 Agent thinking...

🤖 Assistant:
Here are the users from the database:
1. Alice (alice@example.com)
2. Bob (bob@example.com)

────────────────────────────────────────────────────────────

You: (Press Ctrl+C to exit)
```

## Learning Resources

This project demonstrates:
- Tool calling (function calling) with LLMs
- Interactive multi-turn conversations with context
- Streaming responses
- Decorator patterns in Python
- Clean code architecture with separation of concerns
- Type hints and comprehensive documentation
- Environment variable management with python-dotenv
