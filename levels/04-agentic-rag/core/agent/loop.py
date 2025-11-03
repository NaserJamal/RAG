"""
Agent Loop - Core orchestration of the agent's reasoning cycle.

This module implements the main agent loop that:
1. Calls the LLM with available tools
2. Streams responses to the user
3. Detects and routes tool calls
4. Manages conversation flow
"""

from typing import List, Dict, Any
from openai import OpenAI

from core.tool_system import registry
from core.agent.executor import execute_tool_call
from interface.ui import print_thinking, print_assistant_message, print_error, print_separator


def run_agent_loop(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str
) -> None:
    """
    Run the main agent loop with tool calling support.

    Args:
        client: OpenAI client instance
        messages: Conversation history
        model: Model name to use

    The loop continues until the LLM produces a final response
    without requesting any tool calls.
    """
    tools = registry.get_tool_definitions()

    while True:
        print_thinking()

        try:
            # Stream the LLM response
            stream = client.chat.completions.create(
                messages=messages,
                model=model,
                tools=tools,
                stream=True
            )

            # Collect the streaming response
            collected_content = ""
            collected_tool_calls = []
            finish_reason = None

            assistant_header_printed = False

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Update finish reason (preserve non-None values)
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Stream content to user
                if delta.content:
                    if not assistant_header_printed:
                        print_assistant_message("", stream=False)
                        assistant_header_printed = True
                    print_assistant_message(delta.content, stream=True)
                    collected_content += delta.content

                # Collect tool calls
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        # Initialize tool call storage if needed
                        while len(collected_tool_calls) <= tool_call_delta.index:
                            collected_tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })

                        tc = collected_tool_calls[tool_call_delta.index]

                        # Accumulate tool call data
                        if tool_call_delta.id:
                            tc["id"] = tool_call_delta.id
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                tc["function"]["name"] += tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                tc["function"]["arguments"] += tool_call_delta.function.arguments

            # Add newline after streaming content
            if collected_content:
                print()

            # Handle tool calls if any
            if finish_reason == "tool_calls" and collected_tool_calls:
                # Add assistant's tool call message to history
                messages.append({
                    "role": "assistant",
                    "content": collected_content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": tc["type"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        }
                        for tc in collected_tool_calls
                    ]
                })

                # Execute each tool call and add results
                for tc in collected_tool_calls:
                    # Create tool call object
                    class ToolCall:
                        def __init__(self, id, function_name, arguments):
                            self.id = id
                            self.function = type('obj', (object,), {
                                'name': function_name,
                                'arguments': arguments
                            })()

                    tool_call_obj = ToolCall(
                        tc["id"],
                        tc["function"]["name"],
                        tc["function"]["arguments"]
                    )

                    # Execute and get result
                    result = execute_tool_call(tool_call_obj)

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result
                    })

                # Continue loop to get final response
                continue

            else:
                # No tool calls - add final message and exit
                if collected_content:
                    messages.append({
                        "role": "assistant",
                        "content": collected_content
                    })
                break

        except Exception as e:
            print_error(f"{type(e).__name__}: {e}")
            break

    print_separator()
