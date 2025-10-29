from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# ✅ 모델 로드 (TinyLLaMA or fallback)
generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto",
    device_map="auto"
)

@app.route("/")
def home():
    return jsonify({"message": "✅ Flask LLM server is running!"})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    outputs = generator(prompt, max_new_tokens=100, temperature=0.8, top_p=0.95)
    response_text = outputs[0]["generated_text"]
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
