from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for cross-origin requests
import torch
import os
from emotion_recognition import (
    Config,
    EmotionRecognitionModel,
    AudioFeatureExtractor,
    predict_emotion,
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize configuration, model, and feature extractor
config = Config()
feature_extractor = AudioFeatureExtractor(config)
model = EmotionRecognitionModel(config)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Configure upload folder
UPLOAD_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file
        audio_path = os.path.join(UPLOAD_FOLDER, 'temp_audio.wav')
        audio_file.save(audio_path)

        # Predict emotion
        emotion = predict_emotion(audio_path, model, feature_extractor, config)
        
        # Clean up the temporary file
        os.remove(audio_path)
        
        return jsonify({'emotion': emotion}), 200

    except Exception as e:
        # Clean up if file exists
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)