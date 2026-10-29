import { useCallback, useEffect, useRef, useState } from 'react';
import { useI18n } from '../i18n/UiLanguage';
import type { TextKey } from '../i18n/messages';

export function cameraMessage(cause: unknown) {
  const name = cause && typeof cause === 'object' && 'name' in cause ? cause.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return 'cameraDenied';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'cameraMissing';
  if (name === 'NotReadableError') return 'cameraBusy';
  return 'cameraFailure';
}

export function useCamera() {
  const { t } = useI18n();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const generation = useRef(0);
  const [ready, setReady] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<TextKey | null>(null);
  const stop = useCallback(() => {
    generation.current++;
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setReady(false); setStarting(false);
  }, []);
  useEffect(() => () => { generation.current++; streamRef.current?.getTracks().forEach(track => track.stop()); }, []);

  async function start() {
    if (starting || streamRef.current) return;
    const token = ++generation.current;
    setStarting(true); setError(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API unavailable');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
      });
      if (token !== generation.current) { stream.getTracks().forEach(track => track.stop()); return; }
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      stream.getVideoTracks().forEach(track => { track.onended = () => { stop(); setError('cameraDisconnected'); }; });
    } catch (cause) {
      if (token === generation.current) { setError(cameraMessage(cause)); setStarting(false); }
    }
  }

  async function snapshot(): Promise<Blob> {
    const video = videoRef.current;
    if (!video || !ready || !video.videoWidth) throw new Error('La cámara todavía no está lista. Espera un momento.');
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 1280 / video.videoWidth, 720 / video.videoHeight);
    canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
    const context = canvas.getContext('2d');
    if (!context) throw new Error('No se pudo preparar la captura. Intenta nuevamente.');
    // CSS mirrors only the preview. No canvas transform or pixel flip.
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve, reject) => canvas.toBlob(blob => {
      if (blob) resolve(blob); else reject(new Error('No se pudo generar la imagen. Intenta nuevamente.'));
    }, 'image/jpeg', 0.92));
  }

  return { videoRef, ready, starting, error: error ? t(error) : null, start, stop, snapshot,
    onReady: () => { if (streamRef.current) { setReady(true); setStarting(false); } } };
}
