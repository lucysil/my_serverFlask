from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Flask 앱 생성
app = Flask(__name__)
CORS(app)

# 🔹 사용할 모델 (너의 LoRA 기반 LLM)
MODEL_NAME = "Lucysil/my-tinyllama-lora-chatbot"

print("🔹 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("✅ 모델 로드 완료")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # 🔹 모델 추론
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.8,
            top_p=0.95
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": text})
    except Exception as e:
        print("❌ 내부 오류:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ Flask LLM 서버 작동 중!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
