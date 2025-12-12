import asyncio
import os
import sys
import json
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

# Google GenAI Imports
from google import genai
from google.genai import types

# --- Configuration ---
# Load environment variables from .env file if present
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "abis-345004")
DATASET_ID = os.getenv("BQ_DATASET_ID", "gemini_analytics_db")
LOCATION = os.getenv("BQ_LOCATION", "US") # Required by BigQuery MCP
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_ID = "gemini-3-pro-preview" 

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    print("Please set it in your terminal or a .env file.")
    sys.exit(1)

# --- Helper: Convert MCP Tool to Gemini Tool ---
def mcp_tool_to_gemini_decl(mcp_tool: Tool) -> types.FunctionDeclaration:
    """
    Converts an MCP Tool definition into a Gemini FunctionDeclaration.
    """
    return types.FunctionDeclaration(
        name=mcp_tool.name,
        description=mcp_tool.description,
        parameters=mcp_tool.inputSchema 
    )

# --- Main Application Class ---
class BigQueryAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.chat = None
        self.mcp_session: Optional[ClientSession] = None

    async def run_loop(self):
        """Main chat loop."""
        
        # 1. Define Server Parameters (using uvx to run the official MCP BigQuery server)
        # We pass your specific Project and Dataset to restrict the scope.
        server_params = StdioServerParameters(
            command="uvx",
            args=[
                "--quiet", # Suppress uvx logs to prevent stdout pollution
                "mcp-server-bigquery",
                "--project", PROJECT_ID,
                "--location", LOCATION,
                # Filtering by dataset helps keep the context focused for the LLM
                "--dataset", DATASET_ID 
            ],
            env=os.environ.copy() # Pass current env to inherit gcloud auth
        )

        print(f"🔌 Connecting to BigQuery MCP Server...")
        print(f"   Project: {PROJECT_ID}")
        print(f"   Location: {LOCATION}")
        print(f"   Dataset: {DATASET_ID}")

        # 2. Start MCP Connection
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.mcp_session = session
                
                # CRITICAL: Initialize the session explicitly. 
                # This sends the 'initialize' handshake required by the MCP protocol.
                # Without this, the server rejects any subsequent requests (like list_tools).
                print("🤝 Performing MCP Handshake...")
                await session.initialize()
                
                # 3. Initialize Available Tools
                print("🛠️  Fetching tools from MCP server...")
                mcp_tools_list = await session.list_tools()
                
                gemini_tools = [mcp_tool_to_gemini_decl(tool) for tool in mcp_tools_list.tools]
                tool_names = [t.name for t in mcp_tools_list.tools]
                print(f"   Found tools: {', '.join(tool_names)}")

                # 4. Initialize Gemini Chat Session
                # We provide a system instruction to give context about the video analytics use case.
                system_instr = (
                    "You are an expert Data Analyst Agent - your task is to answer user questions about video analytics  "
                    " assume that the users are not familiar with the video analytics tools & teh technical details of the  database" 
                    "You have access to a BigQuery database containing video analytics data. "
                    "Use the available tools to explore tables, inspect schemas, and execute SQL queries to answer user questions. "
                    "Always verify table schemas  by yourself before querying.  "
                    "Don't prompt the user for table name or scheme structure."
                    "When answering, be concise and data-driven."
                )

                self.chat = self.client.chats.create(
                    model=MODEL_ID,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instr,
                        tools=[types.Tool(function_declarations=gemini_tools)],
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True # We will handle execution manually to route via MCP
                        )
                    )
                )

                print("\n✅ Agent Ready! (Type 'quit' to exit)\n")

                # 5. Chat Loop
                while True:
                    try:
                        user_input = input("\n👤 You: ")
                    except EOFError:
                        break
                        
                    if user_input.lower() in ["quit", "exit", "q"]:
                        break

                    await self.process_message(user_input)

    async def process_message(self, message: str):
        """
        Handles the turn: User -> Gemini -> (Tool Call -> MCP -> Result -> Gemini) -> Answer
        """
        # Send user message to Gemini
        response = self.chat.send_message(message)
        
        # Helper loop to handle potentially multiple tool calls in sequence
        while response.function_calls:
            for tool_call in response.function_calls:
                tool_name = tool_call.name
                tool_args = tool_call.args
                
                print(f"🤖 Agent is calling tool: {tool_name}...")
                
                # Execute tool via MCP Session
                try:
                    result: CallToolResult = await self.mcp_session.call_tool(
                        name=tool_name,
                        arguments=tool_args
                    )
                    
                    # Format result for Gemini
                    # MCP returns a list of content (Text or Image). We usually just need the text.
                    tool_output = "\n".join([c.text for c in result.content if hasattr(c, "text")])
                    
                    if result.isError:
                        print(f"   ❌ Tool Error: {tool_output}")
                    else:
                        print(f"   ✅ Tool Result received ({len(tool_output)} chars)")

                except Exception as e:
                    tool_output = f"Error executing tool: {str(e)}"
                    print(f"   ❌ Execution Exception: {str(e)}")

                # Send tool result back to Gemini
                response = self.chat.send_message(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_output}
                    )
                )
        
        # Final text response
        print(f"🤖 Agent: {response.text}")

if __name__ == "__main__":
    # Ensure uv is installed or allow standard python execution if desired
    # For this script, we rely on 'uvx' which comes with 'uv'
    try:
        asyncio.run(BigQueryAgent().run_loop())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nCritical Error: {e}")