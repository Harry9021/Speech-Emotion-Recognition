"""Audio preprocessing and data augmentation pipeline.

The preprocessing steps here are specifically designed to improve emotion
recognition accuracy across diverse English accents (including Indian English):

1. Pre-emphasis filter — compensates for the natural spectral tilt of speech.
   Indian English has distinct high-frequency characteristics (retroflex
   consonants /ʈ, ɖ/, aspirated stops) that this filter helps preserve.

2. Silence trimming — strips leading/trailing silence so feature extraction
   focuses on actual speech. Indian English speakers sometimes have different
   pause patterns which can confuse a model trained only on American English.

3. Amplitude normalization — peak-normalizes to [-1, 1] removing volume
   differences between recording setups and microphones.

4. Data augmentation (training only):
   - Pitch shifting ±2 semitones: Indian English has a wider fundamental
     frequency range and different intonation contours compared to American
     English. Pitch augmentation helps the model generalise across these.
   - Speed perturbation 0.9x–1.1x: Indian English often has a syllable-timed
     rhythm vs. the stress-timed rhythm of American English. Varying speed
     simulates different cadences.
   - Gaussian noise injection: makes the model robust to noisy recordings
     (common in real-world Indian phone/laptop microphones).
"""

import numpy as np
import librosa


def pre_emphasis(signal, coeff=0.97):
    """Apply first-order high-pass filter."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def trim_silence(audio, top_db=25):
    """Remove leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize_amplitude(audio):
    """Peak-normalize to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def augment_pitch(audio, sr, n_steps=None, pitch_range=(-2, 2)):
    if n_steps is None:
        n_steps = np.random.uniform(*pitch_range)
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def augment_speed(audio, rate=None, speed_range=(0.9, 1.1)):
    if rate is None:
        rate = np.random.uniform(*speed_range)
    return librosa.effects.time_stretch(y=audio, rate=rate)


def augment_noise(audio, factor=0.005):
    return audio + factor * np.random.randn(len(audio))


def preprocess_audio(audio, config, augment=False):
    """Full preprocessing pipeline."""
    audio = trim_silence(audio, top_db=config.TRIM_TOP_DB)
    audio = pre_emphasis(audio, coeff=config.PRE_EMPHASIS)
    audio = normalize_amplitude(audio)

    if augment:
        r = np.random.random()
        if r < 0.33:
            audio = augment_pitch(
                audio, config.SAMPLE_RATE, pitch_range=config.AUGMENT_PITCH_RANGE
            )
        elif r < 0.66:
            audio = augment_speed(audio, speed_range=config.AUGMENT_SPEED_RANGE)
        else:
            audio = augment_noise(audio, factor=config.AUGMENT_NOISE_FACTOR)

    return audio
