import { useCamera } from '../hooks/useCamera';
import { StatusMessage } from './StatusMessage';
import { useI18n } from '../i18n/UiLanguage';
import { useShortcuts } from '../hooks/useShortcuts';

export function CameraPanel({ busy, hasSession, onCapture }: {
  busy: boolean; hasSession: boolean; onCapture: (image: () => Promise<Blob>) => void;
}) {
  const camera = useCamera();
  const { t } = useI18n();
  useShortcuts(key => {
    if (key !== 'enter' || busy || !hasSession || !camera.ready) return false;
    onCapture(camera.snapshot); return true;
  });
  return <section className="panel camera-panel" aria-labelledby="camera-title">
    <div className="panel-heading"><div><span className="eyebrow">{t('cameraStep')}</span><h2 id="camera-title">{t('cameraTitle')}</h2></div>
      <span className={`pill ${camera.ready ? 'connected' : ''}`}>{t(camera.ready ? 'cameraActive' : 'cameraOff')}</span>
    </div>
    <div className="camera-stage">
      <video ref={camera.videoRef} className={camera.ready ? 'camera-preview visible' : 'camera-preview'}
        autoPlay muted playsInline onLoadedData={camera.onReady} aria-label={t('preview')} />
      {!camera.ready && <div className="camera-placeholder"><span className="viewfinder" aria-hidden="true">＋</span>
        <h3>{t(camera.starting ? 'waitingCamera' : 'cameraEmpty')}</h3>
        <p>{t('framing')}</p></div>}
      <span className="camera-caption">{t(camera.ready ? 'mirrored' : 'cameraChoice')}</span>
    </div>
    {camera.error && <StatusMessage error>{camera.error}</StatusMessage>}
    <div className="camera-actions"><button className="button secondary" disabled={busy || camera.starting}
      onClick={() => camera.ready ? camera.stop() : void camera.start()}>
      {t(camera.starting ? 'activating' : camera.ready ? 'stopCamera' : 'startCamera')}</button>
      <button className="button primary" disabled={busy || !hasSession || !camera.ready}
        aria-keyshortcuts="Enter" onClick={() => onCapture(camera.snapshot)}>{t('capture')} <kbd aria-hidden="true">Enter</kbd></button></div>
    <p className="helper">{t('deliberate')}</p>
  </section>;
}
