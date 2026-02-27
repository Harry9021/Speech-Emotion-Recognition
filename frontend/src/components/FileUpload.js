import React, { useRef, useState } from 'react';

export default function FileUpload({ onAnalyze, disabled }) {
  const inputRef = useRef(null);
  const [fileName, setFileName] = useState('');

  const handleChange = (e) => {
    const f = e.target.files[0];
    if (f) setFileName(f.name);
  };

  const handleClick = () => {
    const file = inputRef.current?.files[0];
    if (file) onAnalyze(file);
  };

  return (
    <div className="section">
      <label className="file-label" htmlFor="audio-upload">
        {fileName || 'Choose an audio file'}
      </label>
      <input
        id="audio-upload"
        type="file"
        accept="audio/*"
        className="file-input"
        ref={inputRef}
        onChange={handleChange}
      />
      <button
        className="btn primary"
        onClick={handleClick}
        disabled={!fileName || disabled}
      >
        {disabled ? 'Analyzing…' : 'Analyze File'}
      </button>
    </div>
  );
}
