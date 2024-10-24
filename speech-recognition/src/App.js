import React, { useState, useRef } from 'react';
import { ReactMediaRecorder } from 'react-media-recorder';
import axios from 'axios';
import './App.css';

const App = () => {
  const [emotion, setEmotion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [file, setFile] = useState(null);
  const [mediaBlobUrl, setMediaBlobUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setEmotion('');
      setError('');
    }
  };

  const handleUpload = async (audioBlob) => {
    setIsLoading(true);
    setError('');
    
    try {
      const wavBlob = await convertToWav(audioBlob || file);
      const formData = new FormData();
      formData.append('file', wavBlob, 'audio.wav');

      const { data } = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      setEmotion(data.emotion);
    } catch (error) {
      console.error('Error uploading audio:', error);
      setError(error.response?.data?.error || 'Error processing audio');
    } finally {
      setIsLoading(false);
    }
  };

  const convertToWav = async (audioFileOrBlob) => {
    const arrayBuffer = await audioFileOrBlob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const wavBlob = audioBufferToWav(audioBuffer);
    return new Blob([wavBlob], { type: 'audio/wav' });
  };

  const audioBufferToWav = (buffer) => {
    const numberOfChannels = buffer.numberOfChannels;
    const length = buffer.length * numberOfChannels * 2 + 44;
    const result = new DataView(new ArrayBuffer(length));
    let offset = 0;

    const writeString = (str) => {
      for (let i = 0; i < str.length; i++) {
        result.setUint8(offset++, str.charCodeAt(i));
      }
    };
    writeString('RIFF');
    result.setUint32(offset, 36 + buffer.length * 2, true); offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    result.setUint32(offset, 16, true); offset += 4;
    result.setUint16(offset, 1, true); offset += 2;
    result.setUint16(offset, numberOfChannels, true); offset += 2;
    result.setUint32(offset, buffer.sampleRate, true); offset += 4;
    result.setUint32(offset, buffer.sampleRate * 4, true); offset += 4;
    result.setUint16(offset, numberOfChannels * 2, true); offset += 2;
    result.setUint16(offset, 16, true); offset += 2;
    writeString('data');
    result.setUint32(offset, buffer.length * 2, true); offset += 4;

    for (let i = 0; i < buffer.numberOfChannels; i++) {
      const channel = buffer.getChannelData(i);
      for (let j = 0; j < channel.length; j++) {
        const sample = Math.max(-1, Math.min(1, channel[j]));
        result.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }
    return result.buffer;
  };

  return (
    <div className="App">
      <h1>Speech Emotion Recognition Model</h1>

      {/* File Upload Section */}
      <div>
        <input 
          type="file" 
          accept="audio/*" 
          className="Uploader" 
          ref={fileInputRef} 
          onChange={handleFileChange} 
        />
        <button 
          onClick={() => handleUpload()} 
          disabled={!file || isLoading}
        >
          Upload Audio
        </button>
      </div>

      <div style={{ margin: '20px 0' }}>OR</div>

      {/* Recording Section */}
      <ReactMediaRecorder
        audio
        onStop={(blobUrl, blob) => {
          setMediaBlobUrl(blobUrl);
          setFile(blob);
        }}
        render={({ status, startRecording, stopRecording, clearBlobUrl }) => (
          <div>
            <p>Status: {status}</p>
            <button 
              onClick={startRecording} 
              disabled={isLoading}
            >
              Start Recording
            </button>
            <button 
              onClick={stopRecording} 
              disabled={isLoading || status !== 'recording'}
            >
              Stop Recording
            </button>

            {mediaBlobUrl && (
              <>
                <audio src={mediaBlobUrl} controls />
                <button 
                  onClick={() => handleUpload(file)}
                  disabled={isLoading}
                >
                  Upload Recorded Audio
                </button>
                <button
                  onClick={() => {
                    clearBlobUrl();
                    setMediaBlobUrl(null);
                    setFile(null);
                    setEmotion('');
                  }}
                >
                  Re-record
                </button>
              </>
            )}
          </div>
        )}
      />

      {/* Status and Results */}
      {isLoading && <p>Analyzing audio...</p>}
      {error && <p>Error: {error}</p>}
      {emotion && (
        <p>
          Detected Emotion: <strong>{emotion}</strong>
        </p>
      )}
    </div>
  );
};

export default App;