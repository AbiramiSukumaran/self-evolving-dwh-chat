import os
import asyncio
import traceback
import json
import httpx 
import google.auth
import google.auth.transport.requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from types import SimpleNamespace

# --- Standard GenAI SDK ---
from google import genai
from google.genai import types

# --- Logic Imports ---
try:
    from fullagent import analyze_video_and_update_db
except ImportError:
    def analyze_video_and_update_db(*args, **kwargs):
        raise NotImplementedError("fullagent module missing")

# --- Configuration ---
load_dotenv()
app = Flask(__name__)

# Configs
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "abis-345004")
DATASET_ID = os.getenv("BQ_DATASET_ID", "gemini_analytics_db")
LOCATION = os.getenv("BQ_LOCATION", "US") # Required by BigQuery MCP
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-3-pro-preview" 

# Managed MCP Endpoint
MCP_SERVER_URL = "https://bigquery.googleapis.com/mcp"

# --- DIRECT CLIENT IMPLEMENTATION (No ADK) ---
class DirectMcpClient:
    """
    A lightweight JSON-RPC 2.0 client to talk directly to Google's Managed MCP endpoints.
    """
    def __init__(self, base_url, headers):
        self.base_url = base_url
        self.headers = headers
        self.headers['Content-Type'] = 'application/json'
        self.client = httpx.AsyncClient(headers=self.headers, timeout=60.0)

    async def list_tools(self):
        """Sends tools/list JSON-RPC command."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }
        response = await self.client.post(self.base_url, json=payload)
        
        # Enhanced Error Handling
        if response.status_code != 200:
            print(f"DEBUG: MCP List Tools Failed. Status: {response.status_code}")
            print(f"DEBUG: Response Body: {response.text}")
            raise Exception(f"MCP HTTP Error {response.status_code}: {response.text}")

        data = response.json()
        if "error" in data:
            raise Exception(f"MCP Error: {data['error']}")
            
        tools_data = data.get("result", {}).get("tools", [])
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name=t["name"], 
                    description=t.get("description", ""), 
                    inputSchema=t.get("inputSchema", {})
                ) for t in tools_data
            ]
        )

    async def call_tool(self, name, args):
        """Sends tools/call JSON-RPC command."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": args
            },
            "id": 2
        }
        response = await self.client.post(self.base_url, json=payload)
        
        # Enhanced Error Handling
        if response.status_code != 200:
            print(f"DEBUG: MCP Call Tool '{name}' Failed. Status: {response.status_code}")
            print(f"DEBUG: Response Body: {response.text}")
            raise Exception(f"MCP HTTP Error {response.status_code}: {response.text}")

        data = response.json()
        if "error" in data:
            raise Exception(f"MCP Error: {data['error']}")

        content_data = data.get("result", {}).get("content", [])
        return SimpleNamespace(
            content=[
                SimpleNamespace(text=c.get("text", str(c))) 
                for c in content_data
            ]
        )

    async def close(self):
        await self.client.aclose()

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    video_input = None
    is_url = False
    try:
        if 'video' in request.files and request.files['video'].filename != '':
            file = request.files['video']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            video_input = filepath
            is_url = False
        elif 'video_url' in request.form and request.form['video_url'].strip() != '':
            video_input = request.form['video_url']
            is_url = True
        else:
            return jsonify({"error": "No video file or URL provided"}), 400
        
        result = analyze_video_and_update_db(video_input, DATASET_ID, is_url=is_url)
        
        if not is_url and os.path.exists(video_input):
             os.remove(video_input)
        
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
async def chat():
    """
    Direct Approach with Fixed Scopes
    """
    if not GEMINI_API_KEY:
        return jsonify({"response": "Error: GEMINI_API_KEY not configured."}), 500

    data = request.json
    user_message = data.get('message', '')

    direct_client = None
    try:
        # 1. Authenticate with WIDER SCOPES
        # Changed from ['bigquery'] to ['cloud-platform'] to avoid scope permission errors
        creds, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "x-goog-user-project": project_id or PROJECT_ID
        }

        # 2. Initialize Direct Client
        direct_client = DirectMcpClient(MCP_SERVER_URL, headers)

        # 3. Fetch Tools
        mcp_tools = await direct_client.list_tools()
        
        # 4. Convert to Gemini Format
        gemini_tools_declarations = []
        for tool in mcp_tools.tools:
            gemini_tools_declarations.append(types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema
            ))
        
        # 5. Initialize Standard GenAI Client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 6. Create Chat Session
        chat_session = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=
                        "The dataset is {DATASET_ID} and project is {PROJECT_ID}.".format(
                            DATASET_ID=DATASET_ID,
                            PROJECT_ID=PROJECT_ID
                        ) + 
                        "You are an expert Data Analyst Agent - your task is to answer user questions about video analytics  "
                        " assume that the users are not familiar with the video analytics tools & teh technical details of the  database" 
                        "You have access to a BigQuery database containing video analytics data. "
                        "Use the available tools to explore tables, inspect schemas, and execute SQL queries to answer user questions. "
                        "Always verify table schemas  by yourself before querying.  "
                        "Don't prompt the user for table name or scheme structure."
                        "When answering, be concise and data-driven."
                ,tools=[types.Tool(function_declarations=gemini_tools_declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )

        # 7. Send Message & Manual Execution Loop
        response = chat_session.send_message(user_message)
        
        for _ in range(5):
            if not response.function_calls:
                break
            
            parts_for_next_turn = []
            
            for call in response.function_calls:
                print(f"Executing Tool (Direct): {call.name}")
                try:
                    # Execute Tool using Direct Client
                    tool_result = await direct_client.call_tool(call.name, call.args)
                    
                    # Extract Text
                    result_text = "\n".join([c.text for c in tool_result.content])

                    parts_for_next_turn.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": result_text}
                        )
                    )
                except Exception as tool_err:
                    print(f"Tool Error: {tool_err}")
                    # If it's a 403, we want to see the details in the chat output too
                    parts_for_next_turn.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"error": str(tool_err)}
                        )
                    )
            
            if parts_for_next_turn:
                response = chat_session.send_message(parts_for_next_turn)

        return jsonify({"response": response.text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"System Error: {str(e)}"}), 500
        
    finally:
        if direct_client:
            await direct_client.close()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)