import { useEffect, useRef, useState } from 'react';
import type { VisualResource } from '../types/api';
import { StatusMessage } from './StatusMessage';
import { useI18n } from '../i18n/UiLanguage';
import type { TextKey } from '../i18n/messages';
import { useShortcuts } from '../hooks/useShortcuts';
import { usePlayerFocus } from '../hooks/usePlayerFocus';
import { SequenceTimeline } from './SequenceTimeline';
import { getConceptDisplayLabel } from '../presentation/conceptLabels';

export function SequencePlayer({ resources }: { resources: VisualResource[] }) {
  const { t } = useI18n();
  const video = useRef<HTMLVideoElement>(null);
  const container = useRef<HTMLElement>(null);
  const focusButton = useRef<HTMLButtonElement>(null);
  const [focused, setFocused] = useState(false);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [restart, setRestart] = useState(0);
  const [completed, setCompleted] = useState(false);
  const [played, setPlayed] = useState<Set<number>>(() => new Set());
  const [error, setError] = useState<TextKey | null>(null);
  useEffect(() => { setIndex(0); setPlaying(true); setCompleted(false); setPlayed(new Set()); setError(null); }, [resources]);
  usePlayerFocus(focused, container, focusButton);
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
  const go = (next: number, shouldPlay = true) => { setIndex(next); setPlaying(shouldPlay); setCompleted(false); setError(null); };
  const repeat = () => { if (video.current) video.current.currentTime = 0; setPlayed(new Set()); go(0); setRestart(value => value + 1); };
  useShortcuts(key => {
    if (key !== 'p' || resources.length === 0) return false;
    repeat(); return true;
  });
  const current = resources[index];
  if (!current) return null;
  const displayLabel = getConceptDisplayLabel(current.concept_id, current.language);
  return <section ref={container} className={`visual-output ${focused ? 'is-focused' : ''}`} aria-labelledby="visual-title"
    role={focused ? 'dialog' : undefined} aria-modal={focused ? true : undefined}>
    <div className="panel-heading"><div><span className="eyebrow">{t('visualOutput')} · {current.language}</span><h3 id="visual-title">{t('visualTitle')}</h3></div>
      <div className="player-heading-tools"><span className="pill" aria-live="polite">{index + 1} / {resources.length}</span>
        <button ref={focusButton} aria-expanded={focused} onClick={() => setFocused(value => !value)}>
          {t(focused ? 'closeFocusMode' : 'focusMode')}</button></div></div>
    <SequenceTimeline resources={resources} index={index} played={played} completed={completed} onSelect={next => {
      if (video.current && next === index) video.current.currentTime = 0;
      go(next, false);
    }} />
    <video key={`${index}-${current.url}-${restart}`} ref={video} src={current.url} muted playsInline preload="auto"
      aria-label={t('videoLabel', { concept: displayLabel })} onEnded={() => {
        setPlayed(previous => new Set(previous).add(index));
        if (index + 1 < resources.length) go(index + 1);
        else { setPlaying(false); setCompleted(true); }
      }} onError={() => { setPlaying(false); setError('videoFailed'); }} />
    <div className="video-caption"><strong lang={current.language === 'LSM' ? 'es-MX' : 'en-US'}>{displayLabel}</strong>
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
