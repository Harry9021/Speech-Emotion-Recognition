# Speech Emotion Recognition

A production-ready speech emotion recognition system with a **PyTorch CNN backend** and a **React frontend**. Detects seven emotions from audio — with an accent-robust pipeline designed to work well with **Indian English** and other non-native accents.

---

## Project Structure

```
Speech-Emotion-Recognition/
│
├── backend/                        # Python / Flask API
│   ├── app/                        # Core ML package
│   │   ├── __init__.py             # Package exports
│   │   ├── config.py               # Hyperparameters, paths, labels
│   │   ├── preprocessing.py        # Audio preprocessing & augmentation
│   │   ├── feature_extractor.py    # MFCC + delta feature extraction
│   │   ├── model.py                # CNN architecture
│   │   ├── dataset.py              # RAVDESS download & PyTorch Dataset
│   │   └── predict.py              # Inference utilities
│   ├── main.py                     # Flask API entry point
│   ├── train.py                    # Model training script
│   ├── requirements.txt            # Python dependencies
│   ├── .gitignore
│   ├── weights/                    # Trained model weights (gitignored)
│   │   └── best_model.pth
│   └── data/                       # Dataset storage (gitignored)
│       └── dataset/
│
├── frontend/                       # React SPA
│   ├── public/
│   │   ├── index.html
│   │   └── manifest.json
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.js       # File selection & upload
│   │   │   ├── AudioRecorder.js    # Browser microphone recording
│   │   │   └── EmotionResult.js    # Result display with confidence
│   │   ├── hooks/
│   │   │   └── useEmotionAnalysis.js  # API call logic
│   │   ├── utils/
│   │   │   └── wavConverter.js     # Browser audio → WAV conversion
│   │   ├── App.js                  # Root component
│   │   ├── App.css                 # Styles
│   │   ├── App.test.js             # Tests
│   │   ├── index.js                # Entry point
│   │   └── index.css
│   ├── package.json
│   └── .gitignore
│
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | PyTorch |
| Audio Processing | Librosa |
| API | Flask + Flask-CORS |
| Frontend | React 18, Axios, react-media-recorder |
| Data | RAVDESS (auto-downloaded) |

---

## How It Works

### 1. Audio Preprocessing (`preprocessing.py`)

| Step | What it does | Why it matters for Indian English |
|------|-------------|-----------------------------------|
| Silence trimming | Removes dead air | Indian English speakers sometimes use different pause patterns |
| Pre-emphasis filter | Boosts high frequencies (coeff 0.97) | Preserves retroflex consonants (/ʈ, ɖ/) and aspiration patterns |
| Peak normalization | Scales amplitude to [-1, 1] | Removes volume differences across recording setups |

### 2. Feature Extraction (`feature_extractor.py`)

- **13 MFCCs + 13 delta + 13 delta-delta = 39-dimensional feature vector**
- **CMVN** (Cepstral Mean and Variance Normalization) per utterance removes speaker/channel bias

Delta features capture the *rate of change* of spectral features, which correlates with emotion (pitch contour, energy dynamics) and is **more accent-invariant** than raw MFCCs.

### 3. Data Augmentation (`preprocessing.py`, training only)

| Augmentation | Range | Why |
|-------------|-------|-----|
| Pitch shifting | ±2 semitones | Indian English has wider F0 range and different intonation |
| Speed perturbation | 0.9x – 1.1x | Simulates syllable-timed rhythm of Indian English |
| Noise injection | Gaussian, σ=0.005 | Robustness to noisy laptop/phone microphones |

### 4. CNN Classifier (`model.py`)

```
Input (39 × 100) → Conv1D(64) → BN → ReLU → Pool
                  → Conv1D(128) → BN → ReLU → Pool
                  → Conv1D(256) → BN → ReLU → AdaptiveAvgPool(8)
                  → FC(2048→256) → ReLU → Dropout(0.5)
                  → FC(256→128) → ReLU → Dropout(0.3)
                  → FC(128→7)
```

### 5. Detected Emotions

`angry` · `happy` · `sad` · `neutral` · `fearful` · `disgust` · `surprised`

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Harry9021/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

### 2. Backend

```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

Downloads RAVDESS automatically, trains the CNN, saves `weights/best_model.pth`.

### 4. Start the API

```bash
python main.py
# → http://localhost:5000
```

### 5. Frontend

```bash
cd ../frontend
npm install
npm start
# → http://localhost:3000
```

### 6. CLI Prediction (optional)

```bash
cd backend
python -m app.predict path/to/audio.wav
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API & model status |
| `/upload` | POST | Analyse uploaded audio → emotion + confidence |

**POST /upload** response:

```json
{
  "emotion": "happy",
  "confidence": 87.3
}
```

---

## Improving Indian Accent Accuracy

The RAVDESS dataset contains only American English speakers. The preprocessing pipeline makes the model more accent-agnostic, but for **best results** with Indian English:

1. Supplement training data with Indian-accented emotional speech (e.g., IEMOCAP, custom recordings)
2. Retrain with `python train.py`

The augmentation and CMVN normalisation ensure the model focuses on **emotional cues** (pitch contour, energy, speaking rate) rather than accent-specific spectral positions.

---

## Contributing

1. Fork & clone
2. Create a branch (`git checkout -b feature/your-feature`)
3. Commit (`git commit -m "feat: description"`)
4. Push & open a PR

---

Made by [@Harry9021](https://github.com/Harry9021)
