"""Feature extraction: MFCC + delta + delta-delta with CMVN.

Why delta features matter for accent robustness:
  Raw MFCCs capture the spectral envelope at each frame, which is heavily
  influenced by accent-specific formant positions. Delta and delta-delta
  coefficients capture the *rate of change* of these spectral features,
  which correlates more with emotion (pitch contour, energy dynamics) and
  less with accent-specific vowel/consonant placement.

Why CMVN (Cepstral Mean and Variance Normalization):
  Per-utterance CMVN removes speaker-specific and channel-specific bias.
  This is critical when the training data (RAVDESS, American English) differs
  from the deployment accent (Indian English), because CMVN forces the model
  to rely on relative spectral patterns rather than absolute values.
"""

import numpy as np
import librosa

from .preprocessing import preprocess_audio


class AudioFeatureExtractor:
    def __init__(self, config):
        self.config = config

    def load_audio(self, audio_path):
        try:
            audio, _ = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
            return audio
        except Exception as e:
            print(f"[FeatureExtractor] Error loading {audio_path}: {e}")
            return None

    def extract_features(self, audio_path, augment=False):
        """Extract 39-dim features (13 MFCC + 13 Δ + 13 ΔΔ) with CMVN."""
        audio = self.load_audio(audio_path)
        if audio is None:
            return None

        audio = preprocess_audio(audio, self.config, augment=augment)

        try:
            hop = int(self.config.FRAME_STRIDE * self.config.SAMPLE_RATE)
            n_fft = int(self.config.FRAME_LENGTH * self.config.SAMPLE_RATE)

            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.config.SAMPLE_RATE,
                n_mfcc=self.config.N_MFCC,
                hop_length=hop,
                n_fft=n_fft,
            )
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)

            features = np.concatenate([mfccs, delta, delta2], axis=0)

            # CMVN per utterance
            features = (features - np.mean(features, axis=1, keepdims=True)) / (
                np.std(features, axis=1, keepdims=True) + 1e-8
            )

            # Pad / truncate to fixed length
            if features.shape[1] < self.config.TARGET_LENGTH:
                pad = self.config.TARGET_LENGTH - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad)))
            else:
                features = features[:, : self.config.TARGET_LENGTH]

            return features

        except Exception as e:
            print(f"[FeatureExtractor] Error processing {audio_path}: {e}")
            return None
