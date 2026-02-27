import { useState } from 'react';
import axios from 'axios';
import { blobToWav } from '../utils/wavConverter';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export function useEmotionAnalysis() {
  const [emotion, setEmotion] = useState('');
  const [confidence, setConfidence] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const reset = () => {
    setEmotion('');
    setConfidence(null);
    setError('');
  };

  const analyze = async (blob) => {
    if (!blob) return;
    setIsLoading(true);
    reset();

    try {
      const wav = await blobToWav(blob);
      const form = new FormData();
      form.append('file', wav, 'audio.wav');

      const { data } = await axios.post(`${API_URL}/upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setEmotion(data.emotion);
      setConfidence(data.confidence ?? null);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(
        err.response?.data?.error ||
        'Failed to analyse audio. Is the backend running?'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return { emotion, confidence, isLoading, error, analyze, reset };
}
