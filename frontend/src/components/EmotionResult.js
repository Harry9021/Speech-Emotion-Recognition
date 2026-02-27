import React from 'react';

const ICONS = {
  angry: '😠',
  happy: '😊',
  sad: '😢',
  neutral: '😐',
  fearful: '😨',
  disgust: '🤢',
  surprised: '😲',
};

export default function EmotionResult({ emotion, confidence, isLoading, error }) {
  if (isLoading) {
    return (
      <div className="result-card loading">
        <div className="spinner" />
        <p>Analyzing audio…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-card error-card">
        <p>{error}</p>
      </div>
    );
  }

  if (!emotion) return null;

  return (
    <div className="result-card success">
      <span className="emotion-icon">{ICONS[emotion] || '🎵'}</span>
      <h2>{emotion.charAt(0).toUpperCase() + emotion.slice(1)}</h2>
      {confidence !== null && (
        <div className="confidence">
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${confidence}%` }} />
          </div>
          <span>{confidence}% confidence</span>
        </div>
      )}
    </div>
  );
}
