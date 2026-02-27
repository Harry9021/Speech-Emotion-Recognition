import os
import sys
import shutil
import zipfile

import torch
import wget
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import Config

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"


class EmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, feature_extractor, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.augment = augment

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        features = self.feature_extractor.extract_features(
            self.audio_paths[idx], augment=self.augment
        )
        if features is None:
            features = torch.zeros((Config.AUDIO_FEATURE_DIM, Config.TARGET_LENGTH))
        else:
            features = torch.FloatTensor(features)

        return features, self.labels[idx]


def prepare_dataset(dataset_dir=None):
    """Download and organise the RAVDESS dataset."""
    if dataset_dir is None:
        dataset_dir = Config.DATASET_DIR

    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        print("Dataset already exists.")
        return

    os.makedirs(dataset_dir, exist_ok=True)
    tmp = os.path.join(os.path.dirname(dataset_dir), "_tmp_download")
    os.makedirs(tmp, exist_ok=True)

    try:
        print("Downloading RAVDESS…")
        wget.download(RAVDESS_URL, os.path.join(tmp, "ravdess.zip"))
    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)

    print("\nExtracting…")
    with zipfile.ZipFile(os.path.join(tmp, "ravdess.zip"), "r") as z:
        z.extractall(tmp)

    for emo in RAVDESS_EMOTIONS.values():
        os.makedirs(os.path.join(dataset_dir, emo), exist_ok=True)

    print("Organising files…")
    for root, _, files in os.walk(tmp):
        for f in tqdm(files):
            if f.endswith(".wav"):
                code = f.split("-")[2]
                name = RAVDESS_EMOTIONS.get(code)
                if name:
                    shutil.copy2(
                        os.path.join(root, f),
                        os.path.join(dataset_dir, name, f),
                    )

    shutil.rmtree(tmp)
    print("Dataset ready.")
