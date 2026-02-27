import torch.nn as nn


class EmotionRecognitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(config.AUDIO_FEATURE_DIM, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 8, config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.FUSION_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.NUM_EMOTION_CLASSES),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        return self.classifier(x)
