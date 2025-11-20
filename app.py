import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from agent import analyze_video_and_update_db

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    video_input = None
    is_url = False

    try:
        # CASE 1: File Upload
        if 'video' in request.files and request.files['video'].filename != '':
            file = request.files['video']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved locally: {filepath}")
            video_input = filepath
            is_url = False

        # CASE 2: URL Link (Direct Pass-through)
        elif 'video_url' in request.form and request.form['video_url'].strip() != '':
            url = request.form['video_url']
            print(f"Received URL: {url}")
            video_input = url
            is_url = True

        else:
            return jsonify({"error": "No video file or URL provided"}), 400
        
        # Trigger the Gemini Agent
        DATASET_ID = "gemini_analytics_db" 
        
        # We pass 'is_url' so the agent knows how to handle the input string
        result = analyze_video_and_update_db(video_input, DATASET_ID, is_url=is_url)
        
        # Optional: Cleanup local file if it was an upload
        if not is_url and os.path.exists(video_input):
             os.remove(video_input)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get the port from the environment, default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)