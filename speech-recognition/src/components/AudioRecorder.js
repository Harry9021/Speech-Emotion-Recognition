import React, { useState } from 'react';
import { ReactMediaRecorder } from 'react-media-recorder';

export default function AudioRecorder({ onAnalyze, disabled }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [blob, setBlob] = useState(null);

  return (
    <ReactMediaRecorder
      audio
      onStop={(url, b) => { setBlobUrl(url); setBlob(b); }}
      render={({ status, startRecording, stopRecording, clearBlobUrl }) => (
        <div className="section">
          <div className="recording-status">
            <span className={`dot ${status === 'recording' ? 'recording' : ''}`} />
            <span>{status === 'recording' ? 'Recording…' : 'Ready'}</span>
          </div>

          <div className="btn-row">
            <button
              className="btn record"
              onClick={startRecording}
              disabled={disabled || status === 'recording'}
            >
              Record
            </button>
            <button
              className="btn stop"
              onClick={stopRecording}
              disabled={disabled || status !== 'recording'}
            >
              Stop
            </button>
          </div>

          {blobUrl && (
            <div className="playback">
              <audio src={blobUrl} controls />
              <div className="btn-row">
                <button
                  className="btn primary"
                  onClick={() => onAnalyze(blob)}
                  disabled={disabled}
                >
                  {disabled ? 'Analyzing…' : 'Analyze Recording'}
                </button>
                <button
                  className="btn secondary"
                  onClick={() => {
                    clearBlobUrl();
                    setBlobUrl(null);
                    setBlob(null);
                  }}
                >
                  Re-record
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    />
  );
}
