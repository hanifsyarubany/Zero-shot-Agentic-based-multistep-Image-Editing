from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, List
import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from openai import AsyncOpenAI
from io import BytesIO
from PIL import Image
import base64
import json
import re
import os

# Load environment variables & Apply nest_asyncio
nest_asyncio.apply()
load_dotenv(".env")

# Global variables to store session state
session = None
session: ClientSession | None = None
exit_stack: AsyncExitStack | None = None
openai_client = AsyncOpenAI()
session_path = "./session-asset/s1"
model = "gpt-5"
img_index = 1
limit_history_entries = 3

# Tools definition
tools: List[Dict[str, Any]] = []
flagging_tools: List[Dict[str, Any]] = []
routing_tools: List[Dict[str, Any]] = []
editing_tools: List[Dict[str, Any]] = []

# ----- Lifecycle -----
async def bootstrap(url: str = "http://localhost:8050/sse"):
    """Open SSE + MCP ClientSession and populate tool lists."""
    global session, exit_stack, tools, flagging_tools, routing_tools, editing_tools

    exit_stack = AsyncExitStack()
    read_stream, write_stream = await exit_stack.enter_async_context(sse_client(url=url))
    session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()

    tools_result = await session.list_tools()
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in tools_result.tools
    ]
    # Defensive split
    flagging_tools = tools[:1]
    routing_tools = tools[1:2]
    editing_tools = tools[2:]
    return session, flagging_tools, routing_tools, editing_tools

async def shutdown():
    """Close everything cleanly."""
    global exit_stack
    if exit_stack is not None:
        await exit_stack.aclose()
        exit_stack = None

# Optional: retained for callers who want a fresh list later
async def get_mcp_tools() -> List[Dict[str, Any]]:
    global tools, session
    if not tools and session is not None:
        tools_result = await session.list_tools()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in tools_result.tools
        ]
    return tools

