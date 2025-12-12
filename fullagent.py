import os
import time
import json
import re
import uuid
import hashlib
import google.generativeai as genai
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

# --- CONFIGURATION ---
PREFERRED_MODEL = "gemini-3-pro-preview" 
FALLBACK_MODEL = "gemini-3-pro-preview"


PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

# Credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "abis-345004")
DATASET_ID = os.getenv("BQ_DATASET_ID", "gemini_analytics_db")
LOCATION = os.getenv("BQ_LOCATION", "US") # Required by BigQuery MCP
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-3-pro-preview" 


if not os.environ.get("GOOGLE_API_KEY"):
    print("⚠️ WARNING: GOOGLE_API_KEY not found.")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY")) 

bq_client = bigquery.Client(project=PROJECT_ID)

def get_content_hash(input_value, is_url=False):
    """
    Generates a digital fingerprint.
    - If URL: Hashes the URL string.
    - If File: Hashes the file bytes.
    """
    sha256_hash = hashlib.sha256()
    if is_url:
        # Hash the URL string itself as a proxy for content
        sha256_hash.update(input_value.encode('utf-8'))
    else:
        with open(input_value, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_video_already_processed(dataset_id, table_id, file_hash):
    """Checks BigQuery to see if this specific file hash already exists."""
    table_ref = f"{PROJECT_ID}.{dataset_id}.{table_id}"
    
    # Parameterized query to prevent injection
    query = f"SELECT 1 FROM `{table_ref}` WHERE video_file_hash = @hash LIMIT 1"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("hash", "STRING", file_hash)]
    )
    
    try:
        results = bq_client.query(query, job_config=job_config).result()
        for _ in results: return True
        return False
    except Exception:
        # Table likely doesn't exist yet
        return False

def get_industry_benchmarks(dataset_id, table_id, metrics):
    """
    Dynamically queries BigQuery to get AVG and MAX for the metrics found in the current video.
    Returns a dictionary of benchmarks.
    """
    table_ref = f"{PROJECT_ID}.{dataset_id}.{table_id}"
    
    # Filter for only numeric metrics that we can average
    numeric_keys = [k for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    
    if not numeric_keys:
        return {}

    # Construct dynamic SQL
    aggregates = []
    for key in numeric_keys:
        aggregates.append(f"AVG({key}) as avg_{key}")
        aggregates.append(f"MAX({key}) as max_{key}")
    
    query = f"SELECT {', '.join(aggregates)} FROM `{table_ref}`"
    
    try:
        results = bq_client.query(query).result()
        row = next(results)
        
        benchmarks = {}
        for key in numeric_keys:
            benchmarks[key] = {
                "industry_avg": round(row[f"avg_{key}"] or 0, 2),
                "industry_top": round(row[f"max_{key}"] or 0, 2)
            }
        return benchmarks
    except Exception:
        return {}

def generate_strategic_advice(industry, current_metrics, benchmarks):
    """
    Uses Gemini to analyze the gap between current performance and industry benchmarks.
    """
    if not benchmarks:
        return "First entry in this category! Benchmarks established."

    advice_prompt = f"""
    You are a Senior Consultant for the '{industry}' sector.
    
    Here is the performance of the Current Video vs Sector Benchmarks:
    {json.dumps({'current': current_metrics, 'benchmarks': benchmarks}, indent=2)}
    
    1. Compare the current video to the Average and Top performers.
    2. Provide 3 bullet points of actionable advice on how to SUSTAIN strengths or ELEVATE weaknesses.
    3. Be specific to the metrics provided.
    
    Return RAW text (no markdown formatting like ** or ##), just clean paragraphs.
    """
    
    try:
        model = genai.GenerativeModel(FALLBACK_MODEL) 
        response = model.generate_content(advice_prompt)
        return response.text
    except Exception:
        return "Could not generate advice at this time."

def analyze_video_and_update_db(video_input, dataset_id, is_url=False):
    print(f"Processing input: {video_input} (URL mode: {is_url})")
    
    video_session_id = str(uuid.uuid4())
    # 1. Generate Hash for Deduplication
    video_file_hash = get_content_hash(video_input, is_url)
    
    # 2. Prepare Content for Gemini
    gemini_content_part = None
    
    if is_url:
        # Native YouTube Support via File URI
        gemini_content_part = {
            "file_data": {
                "mime_type": "video/mp4", 
                "file_uri": video_input
            }
        }
    else:
        # Standard File Upload
        print("Uploading video to Gemini File API...")
        video_file = genai.upload_file(path=video_input)
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)
        if video_file.state.name == "FAILED": raise ValueError("Video processing failed.")
        gemini_content_part = video_file

    # 3. Analysis Prompt (Updated for Broad Bucketing)
    prompt = """
    Analyze this video content. You are an Intelligent Data Architect.
    
    STEP 1: CLASSIFY the Content into a BROAD Category (Table Name).
    - CRITICAL: Group similar topics together. Do not create niche tables.
    - USE THESE BUCKETS IF APPLICABLE:
      - 'tech_and_software': For tutorials, conferences, coding, AI demos, software reviews, DB tutorials.
      - 'entertainment_performance': For concerts, movies, stand-up, theater, trailers.
      - 'retail_and_business': For store walkthroughs, product unboxing, business meetings, finance.
      - 'education_and_learning': For general lectures, history, science, documentaries (non-tech).
      - 'lifestyle_and_vlog': For travel, daily life, pet videos, cooking.
    
    - ONLY create a new name if it strictly does not fit above. Use snake_case.
    
    STEP 2: Log key events and extract NUMERIC metrics (1-10 scales, counts, percentages).
    - Examples: 'excitement_score', 'crowd_size_est', 'clarity_rating', 'defect_count'.

    Return ONLY valid JSON matching this structure:
    {
      "suggested_table_name": "tech_and_software",
      "events": [
        {
           "timestamp_str": "05:23",
           "event_type": "highlight",
           "description": "Key moment description",
           "custom_metrics": {
               "engagement_score": 8.5,
               "audio_clarity": 9
           }
        }
      ]
    }
    """

    # 4. Generate
    try:
        model = genai.GenerativeModel(PREFERRED_MODEL)
        response = model.generate_content([gemini_content_part, prompt], request_options={"timeout": 600})
    except Exception:
        print("Switching to fallback model...")
        model = genai.GenerativeModel(FALLBACK_MODEL)
        response = model.generate_content([gemini_content_part, prompt], request_options={"timeout": 600})

    # 5. Parse
    json_str = response.text.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        cleaner_str = response.text.replace("```json", "").replace("```", "")
        data = json.loads(cleaner_str)

    # 6. Routing & Deduplication
    raw_table_name = data.get("suggested_table_name", "generic_session_logs")
    table_id = re.sub(r'\W+', '_', raw_table_name).lower()
    
    if is_video_already_processed(dataset_id, table_id, video_file_hash):
        return {
            "status": "skipped_duplicate", 
            "message": "Video already processed.",
            "routed_to_table": table_id
        }

    extracted_events = data.get("events", [])

    # 7. Data Flattening
    flattened_rows = []
    video_aggregate_metrics = {} 
    all_keys_found = set(["timestamp_str", "event_type", "description", "video_session_id", "video_file_hash"])

    for row in extracted_events:
        flat_row = {
            "video_session_id": video_session_id, 
            "video_file_hash": video_file_hash,
            "timestamp_str": row.get("timestamp_str"),
            "event_type": row.get("event_type"),
            "description": row.get("description")
        }
        if "custom_metrics" in row:
            for k, v in row["custom_metrics"].items():
                flat_row[k] = v
                all_keys_found.add(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if k not in video_aggregate_metrics: video_aggregate_metrics[k] = []
                    video_aggregate_metrics[k].append(v)
                    
        flattened_rows.append(flat_row)

    # 8. Insert
    if flattened_rows:
        ensure_table_and_schema(dataset_id, table_id, list(all_keys_found), flattened_rows)

    # 9. Benchmarks & Advice
    my_stats = {k: round(sum(v)/len(v), 2) for k, v in video_aggregate_metrics.items() if v}
    benchmarks = get_industry_benchmarks(dataset_id, table_id, my_stats)
    advice = generate_strategic_advice(raw_table_name, my_stats, benchmarks)

    return {
        "status": "success", 
        "routed_to_table": table_id, 
        "my_stats": my_stats,
        "benchmarks": benchmarks,
        "strategic_advice": advice,
        "events_processed": len(flattened_rows)
    }

def ensure_table_and_schema(dataset_id, table_id, required_keys, data_sample):
    table_ref = f"{PROJECT_ID}.{dataset_id}.{table_id}"
    
    existing_schema_map = {}
    try:
        table = bq_client.get_table(table_ref)
        for field in table.schema: existing_schema_map[field.name] = field.field_type
    except NotFound: pass 

    # Data Cleaning / Type Coercion
    cleaned_data = []
    for row in data_sample:
        new_row = row.copy()
        for k, v in row.items():
            if k in existing_schema_map:
                target_type = existing_schema_map[k]
                if target_type == "INTEGER" and isinstance(v, bool): new_row[k] = 1 if v else 0
                elif target_type == "STRING" and not isinstance(v, str): new_row[k] = str(v)
        cleaned_data.append(new_row)
    data_sample = cleaned_data

    # Infer Schema for NEW columns
    schema_map = {}
    for row in data_sample:
        for k, v in row.items():
            if k not in schema_map:
                if isinstance(v, bool): schema_map[k] = bigquery.SchemaField(k, "BOOLEAN")
                elif isinstance(v, int): schema_map[k] = bigquery.SchemaField(k, "INTEGER")
                elif isinstance(v, float): schema_map[k] = bigquery.SchemaField(k, "FLOAT")
                else: schema_map[k] = bigquery.SchemaField(k, "STRING")

    # Apply Schema Updates
    try:
        table = bq_client.get_table(table_ref)
        new_schema = list(table.schema)
        schema_changed = False
        existing_cols = set(existing_schema_map.keys())
        
        for key in required_keys:
            if key not in existing_cols:
                if key in schema_map:
                    new_schema.append(schema_map[key])
                    schema_changed = True
        
        if schema_changed:
            table.schema = new_schema
            bq_client.update_table(table, ["schema"])
            time.sleep(2) 
            
    except NotFound:
        schema = [schema_map[k] for k in required_keys if k in schema_map]
        table = bigquery.Table(table_ref, schema=schema)
        bq_client.create_table(table)
        time.sleep(2)

    bq_client.insert_rows_json(table_ref, data_sample)
    print(f"Data successfully inserted into {table_id}.")