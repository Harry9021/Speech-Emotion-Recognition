"""Flask API entry point for Speech Emotion Recognition."""

import os
import logging

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from app import Config, AudioFeatureExtractor, predict_emotion, load_model
from train import main as train_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

config = Config()
os.makedirs(config.TEMP_DIR, exist_ok=True)
os.makedirs(config.WEIGHTS_DIR, exist_ok=True)

feature_extractor = AudioFeatureExtractor(config)

weights_path = os.path.join(config.WEIGHTS_DIR, "best_model.pth")

if not os.path.exists(weights_path):
    logger.info("No trained model found — training now (this may take a while)…")
    train_model()

try:
    model = load_model(config=config)
    logger.info("Model loaded from %s", config.WEIGHTS_DIR)
except FileNotFoundError:
    model = None
    logger.error("Training finished but model file not found. Check training output.")

server = Flask(__name__)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
CORS(server, origins=[o.strip() for o in cors_origins.split(",")])


@server.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@server.route("/upload", methods=["POST"])
def upload():
    if model is None:
        return jsonify({"error": "Model not loaded. Train first: python train.py"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio = request.files["file"]
    if audio.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    audio_path = os.path.join(config.TEMP_DIR, "upload.wav")
    try:
        audio.save(audio_path)
        result = predict_emotion(audio_path, model, feature_extractor, config)

        if isinstance(result, tuple):
            emotion, confidence = result
            return jsonify({"emotion": emotion, "confidence": confidence}), 200
        return jsonify({"error": "Could not process audio"}), 422

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    server.run(host="0.0.0.0", port=port, debug=debug)
