# 🎤 Speech Emotion Recognition – Modern CNN & Spectrogram Pipeline 🚀

A state‑of‑the‑art **speech emotion recognition** system that converts audio into spectrograms, analyzes them with a lightweight CNN, and serves real‑time predictions via a Flask API. A fully **responsive React** frontend lets you upload clips and see live emotion inference.

---

## 🛠 Tech Stack

### Backend  
- **Flask** for REST API endpoints
- **PyTorch** for model definition & inference 
- **Librosa** for audio loading & mel‑spectrogram extraction
- **scikit‑learn**, **tqdm**, **wget** for data prep & utilities  

### Frontend  
- **React** (Create React App)  
- **Axios** for HTTP requests  
- **React Router**, **Bootstrap 4**, **AOS** for scroll animations  
- **react‑toastify** for notifications  

---

## 📚 Dataset

We leverage the **RAVDESS** (Ryerson Audio‑Visual Database of Emotional Speech and Song) dataset—24 actors, 8 emotions. The backend auto‑downloads & organizes it from Zenodo before training or inference. 

---

## 🧠 Algorithm & Pipeline

1. **Audio → WAV**  
   Convert any input format to 16 kHz WAV (via Librosa/Pydub).  
2. **Spectrogram Extraction**  
   Compute 13‑dimensional MFCC spectrograms (100‑frame fixed length).  
3. **CNN Classifier**  
   - **Conv1D × 3** (64→128→256 filters) + ReLU + MaxPool  
   - **Fusion FC**: 3072 → 256 → 7 classes  
4. **Training**  
   - Loss: CrossEntropy  
   - Optimizer: Adam (lr = 0.001)  
   - Epochs: 50  
5. **Inference**  
   - `/upload` endpoint returns one of seven emotions:  
     `angry`, `happy`, `sad`, `neutral`, `fearful`, `disgust`, `surprised`.

---

## 📂 Project Structure

```
Speech-Emotion-Recognition/
├── ML_Model/                   # 🔍 Backend: data prep, model, Flask API
│   ├── dataset/                # 🎧 Organized RAVDESS audio
│   ├── emotion_recognition.py  # 🤖 Model, dataset prep & train/predict
│   ├── app.py                  # 🌐 Flask API server
│   └── best_model.pth          # 💾 Trained weights
└── speech-recognition/         # 💻 React frontend (responsive SPA)
    ├── public/                 # 🌐 Static assets
    ├── src/                    # 🧱 Components, pages, styles
    └── package.json            # 📦 Dependencies & scripts
```

---

## 🚀 Quick Start

### 1. Clone & Prepare  
```bash
git clone https://github.com/Harry9021/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

### 2. Backend Setup  
```bash
cd ML_Model
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install flask torch torchvision librosa numpy wget tqdm scikit-learn
```

### 3. Frontend Setup  
```bash
cd ../speech-recognition
npm install
```

### 4. Run Everything  
- **Backend**:  
  ```bash
  cd ../ML_Model
  python app.py
  # → http://localhost:5000
  ```
- **Frontend**:  
  ```bash
  cd ../speech-recognition
  npm start
  # → http://localhost:3000
  ```

Upload an audio clip in the UI and watch the emotion appear! 🎉

---

## 🤝 Contributing

1. Fork & clone  
2. Create a branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "feat: add new feature"`)  
4. Push & open a PR 🚀  

---

Made with ❤️ by [@Harry9021](https://github.com/Harry9021)
