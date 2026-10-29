import type { Recognition } from '../types/api';
import { useI18n } from '../i18n/UiLanguage';
const percent = (value: number) => `${(value * 100).toFixed(1)} %`;

export function RecognitionPanel({ recognition }: { recognition: Recognition | null }) {
  const { t } = useI18n();
  return <div key={recognition ? `${recognition.letter}-${recognition.confidence}` : 'empty'} className={`recognition ${recognition ? 'capture-confirmed' : ''}`} aria-label={t('lastRecognition')}>
    <div className="letter-badge">{recognition && <span className="capture-check" aria-label={t('captureConfirmed')}>✓</span>}{recognition?.letter ?? '—'}</div>
    <div className="recognition-body"><span className="eyebrow">{t('lastPrediction')}</span>
      <p>{recognition ? <>{t('recognizedLetter')} · <strong>{percent(recognition.confidence)}</strong> {t('confidence')}</> : t('nextCapture')}</p>
      {recognition && recognition.top_k.length > 0 && <details><summary>{t('alternatives')}</summary>
        <ul className="candidate-list">{recognition.top_k.map((item, i) => <li key={`${item.label}-${i}`}>
          <strong>{item.label}</strong><span>{percent(item.probability)}</span></li>)}</ul></details>}
    </div>
  </div>;
}
