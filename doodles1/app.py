# app.py
from flask import Flask, render_template, request, jsonify, Response
import requests
import json
import time

app = Flask(__name__)

# Available Ollama models - make sure these are pulled locally
AVAILABLE_MODELS = {
    "llama2": "llama2",
    "llama3": "llama3",
}

OLLAMA_BASE_URL = "http://localhost:11434"

def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    """Get list of models available in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

def chat_ollama(prompt, model="llama2", max_retries=3):
    """Chat with Ollama model"""
    if not check_ollama_status():
        return "⚠️ Ollama service is not running. Please start Ollama first."
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} - Sending request to Ollama...")
            response = requests.post(url, json=payload, timeout=120)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response received")
            
            elif response.status_code == 404:
                return f"❌ Model '{model}' not found. Available models: {', '.join(get_available_models())}"
            
            else:
                error_msg = response.text
                print(f"Ollama API Error: {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"API Error {response.status_code}: {error_msg}"
                
        except requests.exceptions.Timeout:
            print("Request timed out")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return "⏱️ Request timed out. The model might be processing a complex query."
            
        except requests.exceptions.ConnectionError:
            print("Connection error - Ollama might not be running")
            return "🔌 Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return f"❌ Unexpected error: {str(e)}"
    
    return "⚠️ Max retries reached. Please try again later."

def chat_ollama_stream(prompt, model="llama2"):
    """Stream chat response from Ollama"""
    if not check_ollama_status():
        yield "data: " + json.dumps({"error": "Ollama service not running"}) + "\n\n"
        return
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                yield f"data: {json.dumps({'chunk': data['response']})}\n\n"
                            if data.get('done', False):
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"data: {json.dumps({'error': f'API Error {response.status_code}'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    """Check system status"""
    ollama_running = check_ollama_status()
    available_models = get_available_models() if ollama_running else []
    
    return jsonify({
        "ollama_running": ollama_running,
        "available_models": available_models,
        "ollama_url": OLLAMA_BASE_URL
    })

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    selected_model = data.get("model", "llama2")
    
    if not user_message:
        return jsonify({"reply": "Please send a valid message."}), 400
    
    # Validate model
    if selected_model not in AVAILABLE_MODELS.values():
        selected_model = "llama2"  # fallback to default
    
    try:
        print(f"Processing message: {user_message[:50]}... with model: {selected_model}")
        reply = chat_ollama(user_message, model=selected_model)
        return jsonify({"reply": reply, "model_used": selected_model})
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({"reply": f"Server error: {str(e)}"}), 500

@app.route('/chat-stream', methods=['POST'])
def chat_stream_route():
    """Streaming chat endpoint for real-time responses"""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    selected_model = data.get("model", "llama2")
    
    if not user_message:
        return jsonify({"error": "Please send a valid message."}), 400
    
    def generate():
        yield from chat_ollama_stream(user_message, model=selected_model)
    
    return Response(generate(), mimetype='text/event-stream')

# Health check endpoint for load balancers
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Make sure Ollama is running: ollama serve")
    print("Available endpoints:")
    print("  - http://localhost:5000 (main chat)")
    print("  - http://localhost:5000/status (system status)")
    print("  - http://localhost:5000/health (health check)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)