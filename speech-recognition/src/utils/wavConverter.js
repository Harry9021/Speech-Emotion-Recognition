/**
 * Convert any browser-supported audio blob to a mono 16-bit PCM WAV blob.
 * Mixing to mono is important because the backend expects single-channel audio.
 */
export async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  const wav = encodeWav(audioBuffer);
  ctx.close();
  return new Blob([wav], { type: 'audio/wav' });
}

function encodeWav(buffer) {
  const sampleRate = buffer.sampleRate;
  const numChannels = 1;
  const bitsPerSample = 16;

  // Mix to mono
  const left = buffer.getChannelData(0);
  const mono = new Float32Array(left.length);
  mono.set(left);
  if (buffer.numberOfChannels > 1) {
    const right = buffer.getChannelData(1);
    for (let i = 0; i < mono.length; i++) {
      mono[i] = (mono[i] + right[i]) / 2;
    }
  }

  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = mono.length * (bitsPerSample / 8);
  const headerSize = 44;

  const view = new DataView(new ArrayBuffer(headerSize + dataSize));
  let off = 0;

  const str = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(off++, s.charCodeAt(i)); };

  str('RIFF');
  view.setUint32(off, headerSize + dataSize - 8, true); off += 4;
  str('WAVE');
  str('fmt ');
  view.setUint32(off, 16, true); off += 4;            // chunk size
  view.setUint16(off, 1, true); off += 2;              // PCM
  view.setUint16(off, numChannels, true); off += 2;
  view.setUint32(off, sampleRate, true); off += 4;
  view.setUint32(off, byteRate, true); off += 4;
  view.setUint16(off, blockAlign, true); off += 2;
  view.setUint16(off, bitsPerSample, true); off += 2;
  str('data');
  view.setUint32(off, dataSize, true); off += 4;

  for (let i = 0; i < mono.length; i++) {
    const s = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }

  return view.buffer;
}
