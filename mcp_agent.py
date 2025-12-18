"""
Generic MCP Agent for Local LLMs (vLLM)

A minimal agent that:
- Connects to any MCP server (stdio or SSE)
- Translates MCP tools to OpenAI-compatible format
- Works with vLLM or any OpenAI-compatible endpoint
- Runs a simple agent loop

Requirements:
    pip install openai mcp

Usage:
    # Use default config file (mcp_agent_config.json)
    python mcp_agent.py

    # Use custom config file
    python mcp_agent.py --config my_config.json

    # Single query mode
    python mcp_agent.py -q "What files are in /tmp?"
"""

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

# MCP SDK imports
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client


@dataclass
class MCPConnection:
    """Holds an MCP session and its tools."""
    name: str
    session: ClientSession
    tools: list[dict] = field(default_factory=list)


@dataclass
class Config:
    """Agent configuration."""
    # LLM settings
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "mistralai/Devstral-Small-2-24B-Instruct-2512"
    temperature: float = 0.15
    max_tokens: int = 4096

    # Agent settings
    max_steps: int = 30
    system_prompt: str = """You are a helpful assistant with access to tools. Use them when needed to accomplish tasks.

When you have completed the user's request, provide your final answer directly without calling more tools.

Be methodical: understand what tools are available, plan your approach, execute step by step, and verify results."""

    # MCP servers (loaded from config file)
    mcp_servers: list[dict] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from a JSON file with environment variable overrides.

        Environment variables take precedence over config file values:
            LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
            AGENT_MAX_STEPS, AGENT_SYSTEM_PROMPT
        """
        with open(path) as f:
            data = json.load(f)

        llm = data.get("llm", {})
        agent = data.get("agent", {})

        # Environment variables override config file values
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", llm.get("base_url", cls.llm_base_url)),
            llm_api_key=os.getenv("LLM_API_KEY", llm.get("api_key", cls.llm_api_key)),
            llm_model=os.getenv("LLM_MODEL", llm.get("model", cls.llm_model)),
            temperature=float(os.getenv("LLM_TEMPERATURE", llm.get("temperature", cls.temperature))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", llm.get("max_tokens", cls.max_tokens))),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", agent.get("max_steps", cls.max_steps))),
            system_prompt=os.getenv("AGENT_SYSTEM_PROMPT", agent.get("system_prompt", cls.system_prompt)),
            mcp_servers=data.get("mcp_servers", []),
        )


class MCPAgent:
    """Generic agent that connects to MCP servers and uses a local LLM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = AsyncOpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.connections: list[MCPConnection] = []
        self.tool_to_connection: dict[str, MCPConnection] = {}
        self._exit_stack = AsyncExitStack()
    
    async def connect_stdio(self, command: str, name: str | None = None, env: dict[str, str] | None = None):
        """Connect to an MCP server via stdio (runs through shell for env var support)."""
        # Merge parent env with provided env (provided takes precedence)
        server_env = {**os.environ, **(env or {})}
        params = StdioServerParameters(command="sh", args=["-c", command], env=server_env)

        # Use exit stack to properly manage context managers
        streams = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(*streams))
        await session.initialize()

        conn_name = name or f"stdio:{parts[0]}"
        conn = MCPConnection(name=conn_name, session=session)

        await self._load_tools(conn)
        self.connections.append(conn)
        print(f"Connected to {conn_name}: {len(conn.tools)} tools")
    
    async def connect_sse(self, url: str, name: str | None = None):
        """Connect to an MCP server via SSE."""
        # Use exit stack to properly manage context managers
        streams = await self._exit_stack.enter_async_context(sse_client(url))
        session = await self._exit_stack.enter_async_context(ClientSession(*streams))
        await session.initialize()

        conn_name = name or f"sse:{url}"
        conn = MCPConnection(name=conn_name, session=session)

        await self._load_tools(conn)
        self.connections.append(conn)
        print(f"Connected to {conn_name}: {len(conn.tools)} tools")
    
    async def _load_tools(self, conn: MCPConnection):
        """Load tools from an MCP connection."""
        result = await conn.session.list_tools()
        
        for tool in result.tools:
            # Convert MCP tool to OpenAI format
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
                },
            }
            conn.tools.append(openai_tool)
            self.tool_to_connection[tool.name] = conn
    
    def get_all_tools(self) -> list[dict]:
        """Get all tools from all connections in OpenAI format."""
        tools = []
        for conn in self.connections:
            tools.extend(conn.tools)
        return tools
    
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool via its MCP connection."""
        conn = self.tool_to_connection.get(name)
        if not conn:
            return f"Error: Unknown tool '{name}'"
        
        try:
            result = await conn.session.call_tool(name, arguments)
            # Extract text content from result
            if result.content:
                texts = []
                for block in result.content:
                    if hasattr(block, 'text'):
                        texts.append(block.text)
                    elif hasattr(block, 'data'):
                        texts.append(f"[Binary data: {len(block.data)} bytes]")
                return "\n".join(texts) if texts else "Tool returned no content"
            return "Tool returned no content"
        except Exception as e:
            return f"Tool error: {str(e)}"
    
    async def run(self, user_message: str) -> str:
        """Run the agent loop for a user message."""
        tools = self.get_all_tools()
        
        if not tools:
            print("Warning: No tools available")
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        for step in range(self.config.max_steps):
            print(f"\n--- Step {step + 1}/{self.config.max_steps} ---")
            
            # Call LLM
            response = await self.llm.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            msg = response.choices[0].message
            content = msg.content
            tool_calls = msg.tool_calls
            
            # Show assistant response
            if content:
                print(f"Assistant: {content}")
            
            # Add to history
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
            })
            
            # No tool calls = done
            if not tool_calls:
                return content or "Done"
            
            # Execute tools
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                
                print(f"  → {fn_name}({json.dumps(fn_args, indent=2) if fn_args else ''})")
                
                result = await self.call_tool(fn_name, fn_args)
                print(f"  ← {result[:500]}{'...' if len(result) > 500 else ''}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": result,
                })
        
        return "Max steps reached"

    async def run_with_history(
        self,
        user_message: str,
        history: list[dict] | None = None,
        on_tool_call: Callable | None = None,
    ) -> tuple[str, list[dict]]:
        """Run agent with conversation history support and tool call callbacks.

        Args:
            user_message: The user's message
            history: Optional conversation history (will be created if None)
            on_tool_call: Optional async callback for tool events

        Returns:
            (response_text, updated_history)
        """
        tools = self.get_all_tools()

        # Initialize or continue history
        if history is None:
            history = [{"role": "system", "content": self.config.system_prompt}]

        history.append({"role": "user", "content": user_message})
        messages = [msg.copy() for msg in history]

        for step in range(self.config.max_steps):
            response = await self.llm.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            msg = response.choices[0].message
            content = msg.content
            tool_calls = msg.tool_calls

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
            })

            # No tool calls = done
            if not tool_calls:
                history.append({"role": "assistant", "content": content or "Done"})
                return content or "Done", history

            # Execute tools with callbacks
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                # Notify about tool start
                if on_tool_call:
                    await on_tool_call({
                        "type": "tool_start",
                        "name": fn_name,
                        "args": fn_args,
                    })

                result = await self.call_tool(fn_name, fn_args)

                # Notify about tool end
                if on_tool_call:
                    await on_tool_call({
                        "type": "tool_end",
                        "name": fn_name,
                        "result": result[:500] if len(result) > 500 else result,
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": result,
                })

        history.append({"role": "assistant", "content": "Max steps reached"})
        return "Max steps reached", history

    async def run_with_history_streaming(
        self,
        user_message: str,
        history: list[dict] | None = None,
        on_tool_call: Callable | None = None,
        on_token: Callable | None = None,
    ) -> tuple[str, list[dict]]:
        """Run agent with streaming token output for real-time TTS.

        Similar to run_with_history but yields tokens as the LLM generates them,
        enabling real-time text-to-speech synthesis.

        Args:
            user_message: The user's message
            history: Optional conversation history (will be created if None)
            on_tool_call: Optional async callback for tool events
            on_token: Optional async callback for each token: (token: str, is_final: bool)

        Returns:
            (response_text, updated_history)
        """
        tools = self.get_all_tools()

        # Initialize or continue history
        if history is None:
            history = [{"role": "system", "content": self.config.system_prompt}]

        history.append({"role": "user", "content": user_message})
        messages = [msg.copy() for msg in history]

        for step in range(self.config.max_steps):
            # Use streaming API
            stream = await self.llm.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            content = ""
            tool_calls_data: dict[int, dict] = {}  # index -> tool call data

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle content tokens
                if delta.content:
                    content += delta.content
                    if on_token:
                        await on_token(delta.content, False)

                # Handle tool calls (accumulated across chunks)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_calls_data[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_data[idx]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_data[idx]["function"]["arguments"] += tc_delta.function.arguments

            # Signal content complete (if we had content)
            if content and on_token:
                await on_token("", True)

            # Convert tool_calls_data to list
            tool_calls = [tool_calls_data[i] for i in sorted(tool_calls_data.keys())] if tool_calls_data else []

            messages.append({
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls if tool_calls else None,
            })

            # No tool calls = done
            if not tool_calls:
                history.append({"role": "assistant", "content": content or "Done"})
                return content or "Done", history

            # Execute tools with callbacks
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args_str = tc["function"]["arguments"]
                fn_args = json.loads(fn_args_str) if fn_args_str else {}

                # Notify about tool start
                if on_tool_call:
                    await on_tool_call({
                        "type": "tool_start",
                        "name": fn_name,
                        "args": fn_args,
                    })

                result = await self.call_tool(fn_name, fn_args)

                # Notify about tool end
                if on_tool_call:
                    await on_tool_call({
                        "type": "tool_end",
                        "name": fn_name,
                        "result": result[:500] if len(result) > 500 else result,
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": result,
                })

        history.append({"role": "assistant", "content": "Max steps reached"})
        return "Max steps reached", history

    async def cleanup(self):
        """Close all MCP connections."""
        await self._exit_stack.aclose()


async def interactive_loop(agent: MCPAgent):
    """Run an interactive chat loop."""
    print("\n" + "=" * 50)
    print("MCP Agent Ready")
    print("=" * 50)
    
    tools = agent.get_all_tools()
    print(f"\nConnected to {len(agent.connections)} MCP server(s)")
    print(f"Available tools ({len(tools)}):")
    for t in tools:
        name = t["function"]["name"]
        desc = t["function"]["description"][:60] + "..." if len(t["function"]["description"]) > 60 else t["function"]["description"]
        print(f"  • {name}: {desc}")
    
    print("\nType your message (or 'quit' to exit, 'tools' to list tools):\n")
    
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "tools":
            for t in tools:
                print(f"\n{t['function']['name']}:")
                print(f"  {t['function']['description']}")
                print(f"  Parameters: {json.dumps(t['function']['parameters'], indent=4)}")
            continue
        
        try:
            result = await agent.run(user_input)
            print(f"\n{'=' * 50}")
            print(f"Result: {result}")
            print('=' * 50)
        except Exception as e:
            print(f"Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Generic MCP Agent for Local LLMs")

    parser.add_argument(
        "--config", "-f",
        default="mcp_agent_config.json",
        help="Path to config file (default: mcp_agent_config.json)",
    )
    parser.add_argument(
        "--query", "-q",
        help="Single query (non-interactive mode)",
    )

    args = parser.parse_args()

    # Load config from file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("\nCreate a config file or specify one with --config")
        sys.exit(1)

    config = Config.from_file(config_path)

    if not config.mcp_servers:
        print("No MCP servers defined in config file")
        sys.exit(1)

    agent = MCPAgent(config)

    try:
        # Connect to all MCP servers from config
        for server in config.mcp_servers:
            server_type = server.get("type", "stdio")
            name = server.get("name")

            if server_type == "stdio":
                command = server.get("command")
                if command:
                    await agent.connect_stdio(command, name)
            elif server_type == "sse":
                url = server.get("url")
                if url:
                    await agent.connect_sse(url, name)

        # Run
        if args.query:
            result = await agent.run(args.query)
            print(f"\nResult: {result}")
        else:
            await interactive_loop(agent)

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

