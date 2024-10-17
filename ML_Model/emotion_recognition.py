import os
import sys
import zipfile
import wget
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Config:
    # Audio processing parameters
    SAMPLE_RATE = 16000
    FRAME_LENGTH = 0.025
    FRAME_STRIDE = 0.010
    N_MFCC = 13
    
    # Model parameters
    AUDIO_FEATURE_DIM = 13
    FUSION_HIDDEN_DIM = 256
    NUM_EMOTION_CLASSES = 7
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioFeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def convert_audio_to_wav(self, audio_path):
        """Convert any audio format to WAV with proper sampling rate."""
        try:
            # Get the file extension
            _, ext = os.path.splitext(audio_path)
            ext = ext.lower()

            if ext == '.wav':
                # If it's already a WAV file, just load it
                audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE)
                return audio
            
            # Create a temporary directory if it doesn't exist
            os.makedirs('temp_audio', exist_ok=True)
            temp_wav_path = os.path.join('temp_audio', 'temp.wav')

            # Convert audio file to WAV using pydub
            if ext in ['.mp3', '.m4a', '.ogg', '.flac']:
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(self.config.SAMPLE_RATE)
                audio.export(temp_wav_path, format='wav')
                
                # Load the converted audio file
                audio, sr = librosa.load(temp_wav_path, sr=self.config.SAMPLE_RATE)
                
                # Clean up
                os.remove(temp_wav_path)
                
                return audio
            else:
                raise ValueError(f"Unsupported audio format: {ext}")
                
        except Exception as e:
            print(f"Error converting audio file: {str(e)}")
            return None
        
    def extract_mfcc(self, audio_path):
        try:
            # Convert audio to WAV format first
            audio = self.convert_audio_to_wav(audio_path)
            if audio is None:
                return None
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.config.SAMPLE_RATE,
                n_mfcc=self.config.N_MFCC,
                hop_length=int(self.config.FRAME_STRIDE * self.config.SAMPLE_RATE),
                n_fft=int(self.config.FRAME_LENGTH * self.config.SAMPLE_RATE)
            )
            
            # Normalize features
            mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
            
            # Pad or truncate to fixed length
            target_length = 100
            if mfccs.shape[1] < target_length:
                pad_width = target_length - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
            else:
                mfccs = mfccs[:, :target_length]
            
            return mfccs
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

class EmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, feature_extractor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        mfccs = self.feature_extractor.extract_mfcc(audio_path)
        if mfccs is None:
            mfccs = torch.zeros((Config.N_MFCC, 100))
        else:
            mfccs = torch.FloatTensor(mfccs)
        
        return mfccs, label

class EmotionRecognitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Audio feature processing
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(self.config.AUDIO_FEATURE_DIM, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Fusion layers
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 * 12, self.config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.config.FUSION_HIDDEN_DIM, self.config.NUM_EMOTION_CLASSES)
        )
        
    def forward(self, audio_features):
        # Process audio features
        audio_out = self.audio_cnn(audio_features)
        audio_out = audio_out.flatten(1)
        
        # Fusion and classification
        fusion_out = self.fusion_fc(audio_out)
        
        return fusion_out

def prepare_dataset():
    if os.path.exists("dataset") and len(os.listdir("dataset")) > 0:
        print("Dataset already exists!")
        return
    
    print("Preparing dataset...")
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    
    # Download RAVDESS dataset
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    try:
        print("Downloading dataset...")
        wget.download(url, "temp/ravdess.zip")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        sys.exit(1)

    print("\nExtracting files...")
    with zipfile.ZipFile("temp/ravdess.zip", 'r') as zip_ref:
        zip_ref.extractall("temp")
    
    emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    
    for emotion in emotions.values():
        os.makedirs(f"dataset/{emotion}", exist_ok=True)
    
    print("Organizing files...")
    for root, dirs, files in os.walk("temp"):
        for file in tqdm(files):
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                emotion_name = emotions.get(emotion_code)
                
                if emotion_name:
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join("dataset", emotion_name, file)
                    shutil.copy2(source_path, dest_path)
    
    print("Cleaning up...")
    shutil.rmtree("temp")
    
    print("Dataset preparation completed!")

def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    best_val_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}'):
            batch_features = batch_features.to(config.DEVICE)
            batch_labels = batch_labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(config.DEVICE)
                batch_labels = batch_labels.to(config.DEVICE)
                
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

def predict_emotion(audio_path, model, feature_extractor, config):
    """Predict emotion from audio file."""
    emotion_map = {
        0: "angry",
        1: "happy",
        2: "sad",
        3: "neutral",
        4: "fearful",
        5: "disgust",
        6: "surprised"
    }
    
    # Extract features
    print(f"Processing audio file: {audio_path}")
    mfccs = feature_extractor.extract_mfcc(audio_path)
    if mfccs is None:
        return "Error processing audio file"
    
    # Prepare input
    mfccs = torch.FloatTensor(mfccs).unsqueeze(0)
    
    # Move to device
    mfccs = mfccs.to(config.DEVICE)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(mfccs)
        _, predicted = torch.max(outputs, 1)
        
    return emotion_map[predicted.item()]

def main():
    # First, prepare the dataset
    prepare_dataset()
    
    # Initialize configuration
    config = Config()
    
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor(config)
    
    # Get mode from user
    mode = input("Enter 1 to 'train' to train a new model or 2 to 'predict' to use existing model: ")
    
    if mode == "1":
        print("Preparing training data...")
        
        # Prepare data
        audio_paths = []
        labels = []
        emotion_to_label = {
            "angry": 0, "happy": 1, "sad": 2, "neutral": 3,
            "fearful": 4, "disgust": 5, "surprised": 6
        }
        
        # Walk through dataset directory
        for emotion in emotion_to_label.keys():
            emotion_dir = os.path.join("dataset", emotion)
            if os.path.exists(emotion_dir):
                for filename in os.listdir(emotion_dir):
                    if filename.endswith('.wav'):
                        audio_paths.append(os.path.join(emotion_dir, filename))
                        labels.append(emotion_to_label[emotion])
        
        # Split dataset
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            audio_paths, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = EmotionDataset(train_paths, train_labels, feature_extractor)
        val_dataset = EmotionDataset(val_paths, val_labels, feature_extractor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
        
        # Initialize model
        model = EmotionRecognitionModel(config).to(config.DEVICE)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Train model
        train_model(model, train_loader, val_loader, criterion, optimizer, config)
        print("Training completed! Model saved as 'best_model.pth'")
        
    elif mode == "2":
        # Load model
        model = EmotionRecognitionModel(config).to(config.DEVICE)
        try:
            model.load_state_dict(torch.load('best_model.pth'))
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Error: Model file 'best_model.pth' not found. Please train the model first.")
            return
        
        # Get audio file for prediction
        # audio_path = input("Enter the path to the audio file for emotion prediction: ")
        # audio_path = "03-01-05-01-01-01-01.wav"
        # if not os.path.exists(audio_path):
        #     print("Error: Audio file not found!")
        #     return
        
        # Make prediction
        emotion = predict_emotion("1.wav", model, feature_extractor, config)
        print(f"Predicted emotion: {emotion}")
    
    else:
        print("Invalid mode selected. Please choose 'train' or 'predict'.")

if __name__ == "__main__":
    main()