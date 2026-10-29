import { useEffect, useRef, useState } from 'react';
import type { VisualResource } from '../types/api';
import { StatusMessage } from './StatusMessage';
import { useI18n } from '../i18n/UiLanguage';
import type { TextKey } from '../i18n/messages';
import { useShortcuts } from '../hooks/useShortcuts';

export function SequencePlayer({ resources }: { resources: VisualResource[] }) {
  const { t } = useI18n();
  const video = useRef<HTMLVideoElement>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [restart, setRestart] = useState(0);
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState<TextKey | null>(null);
  useEffect(() => { setIndex(0); setPlaying(true); setCompleted(false); setError(null); }, [resources]);
  useEffect(() => {
    const element = video.current;
    if (!element) return;
    let cancelled = false;
    if (playing) void element.play().catch((cause: unknown) => {
      if (cancelled || (cause instanceof DOMException && cause.name === 'AbortError')) return;
      setPlaying(false); setError('autoplayFailed');
    });
    else element.pause();
    return () => { cancelled = true; };
  }, [playing, index, resources, restart]);
  const go = (next: number) => { setIndex(next); setPlaying(true); setCompleted(false); setError(null); };
  const repeat = () => { if (video.current) video.current.currentTime = 0; go(0); setRestart(value => value + 1); };
  useShortcuts(key => {
    if (key !== 'p' || resources.length === 0) return false;
    repeat(); return true;
  });
  const current = resources[index];
  if (!current) return null;
  return <section className="visual-output" aria-labelledby="visual-title">
    <div className="panel-heading"><div><span className="eyebrow">{t('visualOutput')} · {current.language}</span><h3 id="visual-title">{t('visualTitle')}</h3></div>
      <span className="pill" aria-live="polite">{index + 1} / {resources.length}</span></div>
    <video key={`${index}-${current.url}-${restart}`} ref={video} src={current.url} muted playsInline preload="auto"
      aria-label={t('videoLabel', { concept: current.concept_id })} onEnded={() => {
        if (index + 1 < resources.length) go(index + 1);
        else { setPlaying(false); setCompleted(true); }
      }} onError={() => { setPlaying(false); setError('videoFailed'); }} />
    <div className="video-caption"><strong>{current.concept_id}</strong>
      <span>{t(completed ? 'sequenceDone' : 'mutedVideos')}</span></div>
    {error && <StatusMessage error>{t(error)}</StatusMessage>}
    <div className="player-controls"><button disabled={index === 0} onClick={() => go(index - 1)}>{t('previous')}</button>
      <button className="button primary" onClick={() => {
        setError(null);
        if (completed) repeat(); else setPlaying(value => !value);
      }}>{t(playing ? 'pause' : 'play')}</button>
      <button disabled={index === resources.length - 1} onClick={() => go(index + 1)}>{t('next')}</button>
      <button aria-keyshortcuts="P" onClick={repeat}>{t('repeat')} <kbd aria-hidden="true">P</kbd></button></div>
  </section>;
}
