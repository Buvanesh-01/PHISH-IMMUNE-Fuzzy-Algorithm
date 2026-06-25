from flask import Flask, request, jsonify, render_template
import os
from predict import load_models, predict_url

app = Flask(__name__, template_folder='.')

# Load models into memory ONCE at startup to ensure real-time speed [cite: 352]
print("--- Initializing Phish-Immune Systems ---")
models, feature_cols = load_models()

@app.route('/')
def home():
    # Serves your console prototype
    return render_template('phishing_console.html')

@app.route('/api/scan', methods=['POST'])
def scan_url():
    data = request.get_json()
    url_to_scan = data.get('url')
    
    if not url_to_scan:
        return jsonify({"error": "No URL provided"}), 400

    # Executes the integrated pipeline: Extraction -> Tier-1 -> Tier-2 [cite: 352]
    result = predict_url(url_to_scan, models, feature_cols, verbose=False)
    
    return jsonify(result)

if __name__ == '__main__':
    # Starts the local server
    app.run(debug=True, port=5000)