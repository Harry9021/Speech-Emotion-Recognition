import os
import torch

from .config import Config
from .feature_extractor import AudioFeatureExtractor
from .model import EmotionRecognitionModel


def predict_emotion(audio_path, model, feature_extractor, config):
    """Return (emotion_label, confidence_pct) or 'error'."""
    features = feature_extractor.extract_features(audio_path, augment=False)
    if features is None:
        return "error"

    features = torch.FloatTensor(features).unsqueeze(0).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    emotion = config.LABEL_TO_EMOTION[predicted.item()]
    return emotion, round(confidence.item() * 100, 1)


def load_model(weights_path=None, config=None):
    """Load trained weights from disk."""
    if config is None:
        config = Config()
    if weights_path is None:
        weights_path = os.path.join(config.WEIGHTS_DIR, "best_model.pth")

    model = EmotionRecognitionModel(config).to(config.DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
    model.eval()
    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.predict <audio_file>")
        sys.exit(1)

    cfg = Config()
    mdl = load_model(config=cfg)
    ext = AudioFeatureExtractor(cfg)

    result = predict_emotion(sys.argv[1], mdl, ext, cfg)
    if isinstance(result, tuple):
        emotion, conf = result
        print(f"Emotion: {emotion} ({conf}% confidence)")
    else:
        print(f"Result: {result}")
