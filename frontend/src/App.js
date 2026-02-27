import React from 'react';
import FileUpload from './components/FileUpload';
import AudioRecorder from './components/AudioRecorder';
import EmotionResult from './components/EmotionResult';
import { useEmotionAnalysis } from './hooks/useEmotionAnalysis';
import './App.css';

export default function App() {
  const { emotion, confidence, isLoading, error, analyze, reset } = useEmotionAnalysis();

  const handleAnalyze = (blob) => {
    reset();
    analyze(blob);
  };

  return (
    <div className="App">
      <h1>Speech Emotion Recognition</h1>
      <p className="subtitle">Upload or record audio to detect emotion</p>

      <FileUpload onAnalyze={handleAnalyze} disabled={isLoading} />

      <div className="divider"><span>OR</span></div>

      <AudioRecorder onAnalyze={handleAnalyze} disabled={isLoading} />

      <EmotionResult
        emotion={emotion}
        confidence={confidence}
        isLoading={isLoading}
        error={error}
      />
    </div>
  );
}
