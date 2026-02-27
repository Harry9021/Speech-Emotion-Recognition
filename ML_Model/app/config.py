import os
import torch


class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
    TEMP_DIR = os.path.join(BASE_DIR, "tmp")

    # Audio processing
    SAMPLE_RATE = 16000
    FRAME_LENGTH = 0.025
    FRAME_STRIDE = 0.010
    N_MFCC = 13
    N_MELS = 40
    TARGET_LENGTH = 100

    # Pre-emphasis (boosts high-freq detail lost in varied recording setups)
    PRE_EMPHASIS = 0.97

    # Silence trimming
    TRIM_TOP_DB = 25

    # Model architecture
    AUDIO_FEATURE_DIM = 39  # 13 MFCC + 13 delta + 13 delta-delta
    FUSION_HIDDEN_DIM = 256
    NUM_EMOTION_CLASSES = 7

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50

    # Augmentation (improves robustness to accents and recording conditions)
    AUGMENT_PITCH_RANGE = (-2, 2)   # semitones
    AUGMENT_SPEED_RANGE = (0.9, 1.1)
    AUGMENT_NOISE_FACTOR = 0.005

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMOTION_LABELS = [
        "angry", "happy", "sad", "neutral",
        "fearful", "disgust", "surprised",
    ]
    EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTION_LABELS)}
    LABEL_TO_EMOTION = {i: e for i, e in enumerate(EMOTION_LABELS)}
