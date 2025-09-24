#!/usr/bin/env python3
"""
Minimal MCP Server Test
Test basic MCP functionality without complex dependencies
"""

import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

class MinimalMCP:
    def __init__(self):
        self.server = Server("minimal-test")
        self._register_tools()
    
    def _register_tools(self):
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="test_tool",
                    description="Simple test tool",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Test message"
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "test_tool":
                message = arguments.get("message", "Hello from minimal MCP!")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": message,
                        "server": "minimal-test"
                    }, indent=2)
                )]
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

async def main():
    mcp = MinimalMCP()
    await mcp.run()

if __name__ == "__main__":
    asyncio.run(main())
