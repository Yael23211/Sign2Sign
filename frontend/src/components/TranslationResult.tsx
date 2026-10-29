import { useEffect, useRef, useState } from 'react';
import type { Translation } from '../types/api';
import { SequencePlayer } from './SequencePlayer';
import { StatusMessage } from './StatusMessage';
import { useI18n } from '../i18n/UiLanguage';
import type { TextKey } from '../i18n/messages';

function TextCard({ title, text, lang }: { title: string; text: string; lang: 'en-US' | 'es-MX' }) {
  const { t } = useI18n();
  const card = useRef<HTMLElement>(null);
  const [notice, setNotice] = useState<TextKey | null>(null);
  async function copy() {
    try { await navigator.clipboard.writeText(text); setNotice('copied'); }
    catch { setNotice('copyFailed'); }
  }
  function speak() {
    if (!window.speechSynthesis || typeof window.SpeechSynthesisUtterance !== 'function') {
      setNotice('speechUnsupported'); return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    const voice = window.speechSynthesis.getVoices().find(item => item.localService && item.lang === lang);
    if (voice) utterance.voice = voice;
    utterance.onerror = event => {
      if (event.error !== 'canceled' && event.error !== 'interrupted') setNotice('speechFailed');
    };
    window.speechSynthesis.speak(utterance);
    setNotice('speechStarted');
  }
  function print() {
    card.current?.classList.add('print-target');
    try { window.print(); } finally { card.current?.classList.remove('print-target'); }
  }
  return <article ref={card} className="text-result"><span className="eyebrow">{title}</span>
    <p className="result-text" lang={lang}>{text || t('noContent')}</p>
    {text && <div className="text-tools"><button onClick={() => void copy()} aria-label={t('copyLabel', { title: title.toLowerCase() })}>{t('copy')}</button>
      <button onClick={speak} aria-label={t('listenLabel', { title: title.toLowerCase() })}>{t('listen')}</button>
      <button onClick={print} aria-label={t('printLabel', { title: title.toLowerCase() })}>{t('print')}</button></div>}
    <span className="tool-notice" role="status">{notice ? t(notice) : null}</span>
  </article>;
}

export function TranslationResult({ result }: { result: Translation | null }) {
  const { t } = useI18n();
  const resultRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (result?.state === 'ready') resultRef.current?.scrollIntoView?.({
      behavior: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start',
    });
  }, [result]);
  useEffect(() => () => { window.speechSynthesis?.cancel(); }, [result]);
  if (!result) return <section className="panel result-placeholder" aria-label={t('resultPending')}>
    <span className="eyebrow">{t('resultStep')}</span><h2>{t('resultEmptyTitle')}</h2>
    <p>{t('resultEmpty')}</p><span aria-hidden="true">ASL ↔ LSM</span></section>;
  const sourceLang = result.source_language === 'ASL' ? 'en-US' : 'es-MX';
  const targetLang = result.target_language === 'ASL' ? 'en-US' : 'es-MX';
  return <section ref={resultRef} className="panel result-panel" aria-labelledby="result-title">
    <div className="panel-heading"><div><span className="eyebrow">{t('resultStep')}</span><h2 id="result-title">{t('resultTitle')}</h2></div>
      <span className="pill">{result.source_language} → {result.target_language}</span></div>
    <div className="recognized-result"><span className="eyebrow">{t('recognizedText')}</span><p>{result.recognized_text}</p></div>
    <div className="text-results"><TextCard title={t('corrected')} text={result.corrected_text} lang={sourceLang} />
      <TextCard title={t('translation')} text={result.translated_text} lang={targetLang} /></div>
    {result.error && <StatusMessage error>{t('visualError', { stage: result.error.stage })}</StatusMessage>}
    {result.unresolved.length > 0 && <StatusMessage>
      {t(result.concepts.length === 0 ? 'unresolvedOnly' : 'unresolvedPartial')}{' '}
      <strong>{t('unresolved')}</strong> {result.unresolved.join(', ')}.</StatusMessage>}
    {result.resources.length > 0 && <SequencePlayer resources={result.resources} />}
    <details className="translation-details"><summary>{t('details')}</summary>
      <p>{t('concepts')}: {result.concepts.join(' · ') || t('none')}</p>
      <p>{t('latency', { seconds: result.latency_seconds.toFixed(2), state: result.state })}</p></details>
  </section>;
}
