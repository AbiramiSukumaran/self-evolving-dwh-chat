Self-Evolving Data Warehouse Chat 🤖

This project is Part 2 of the Agentic Data Engineering series. It implements a conversational analytics agent that connects to a "Self-Evolving" BigQuery Data Warehouse.

Powered by Gemini 3.0 Pro and Google's Managed Model Context Protocol (MCP), this app allows users to ask natural language questions about their video analytics data without pre-defined SQL queries.

🚀 Key Features

Managed MCP Integration: Connects to Google's BigQuery Managed MCP endpoint using a custom "Direct Client" (JSON-RPC over HTTP).

Gemini 3.0 Pro: Uses the latest model for advanced SQL generation and reasoning.

Conversational Memory: Maintains context across chat turns for deep-dive analysis.

Secure Auth: Uses Google Application Default Credentials (ADC) with IAM integration.

🛠️ Architecture

Frontend: Simple HTML/JS Chat Interface.

Backend: Flask App acting as the bridge.

Transport: DirectMcpClient (bypassing heavy SDKs) to speak JSON-RPC 2.0 to bigquery.googleapis.com/mcp.

📦 Setup

Clone the repo:

git clone [https://github.com/AbiramiSukumaran/self-evolving-dwh-chat.git](https://github.com/AbiramiSukumaran/self-evolving-dwh-chat.git)
cd self-evolving-dwh-chat


Install Dependencies:

pip install -r requirements.txt


Environment Variables:
Create a .env file:

GOOGLE_API_KEY=your_key
GCP_PROJECT_ID=your_project
BQ_DATASET_ID=your_dataset
BQ_LOCATION=US
GEMINI_API_KEY=your_gemini_key


Run Locally:

python app.py


☁️ Deployment (Cloud Run)

gcloud run deploy video-analytics-gemini-3 \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=<>,BQ_LOCATION=us-central1,GOOGLE_CLOUD_PROJECT=<>,BQ_DATASET_ID=<>,MODEL_ID=gemini-3-pro-preview,GEMINI_API_KEY=<>,GOOGLE_API_KEY=<>
