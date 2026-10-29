import { useEffect, useRef, useState } from 'react';
import { api, ApiError, errorMessage } from '../api/sign2signApi';
import type { Command, Language, Recognition, Session, Translation } from '../types/api';
import type { TextKey } from '../i18n/messages';

export type Phase = 'idle' | 'session' | 'capturing' | 'command' | 'translating';

export function useSign2SignSession() {
  const [language, setLanguage] = useState<Language>('ASL');
  const [session, setSession] = useState<Session | null>(null);
  const [recognition, setRecognition] = useState<Recognition | null>(null);
  const [result, setResult] = useState<Translation | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [message, setMessage] = useState<{ key: TextKey; values?: Record<string, string> } | null>({ key: 'prompt' });
  const [error, setError] = useState<unknown>(null);
  const active = useRef<Session | null>(null);
  const locked = useRef(false);
  const mounted = useRef(true);
  const update = (next: Session | null) => { active.current = next; if (mounted.current) setSession(next); };

  useEffect(() => {
    mounted.current = true;
    const release = () => {
      const id = active.current?.id;
      if (id) void api.delete(id, true).catch(() => {});
    };
    window.addEventListener('pagehide', release);
    return () => { mounted.current = false; window.removeEventListener('pagehide', release); release(); };
  }, []);

  async function run(nextPhase: Phase, action: () => Promise<void>) {
    if (locked.current) return;
    locked.current = true; setPhase(nextPhase); setError(null); setMessage(null);
    try { await action(); }
    catch (cause) {
      if (mounted.current) {
        setError(cause);
        if (cause instanceof ApiError && cause.status === 404) update(null);
      }
    } finally {
      locked.current = false;
      if (mounted.current) setPhase('idle');
    }
  }

  const start = (source: Language) => run('session', async () => {
    if (active.current) {
      try { await api.delete(active.current.id); }
      catch (cause) { if (!(cause instanceof ApiError && cause.status === 404)) throw cause; }
      update(null);
    }
    if (!mounted.current) return;
    setResult(null); setRecognition(null);
    const created = await api.create(source);
    if (!mounted.current) { void api.delete(created.id, true).catch(() => {}); return; }
    update(created); setLanguage(created.source_language); setMessage({ key: 'sessionReady' });
  });

  const capture = (makeImage: () => Promise<Blob>) => run('capturing', async () => {
    if (!active.current) return;
    const response = await api.capture(active.current.id, await makeImage());
    if (!mounted.current) return;
    update(response.session); setRecognition(response.recognition); setResult(response.session.last_result);
    setMessage(response.recognition ? { key: 'captureAccepted', values: { letter: response.recognition.letter } } : { key: 'noSign' });
  });

  const command = (value: Command) => run('command', async () => {
    if (!active.current) return;
    const next = await api.command(active.current.id, value);
    if (!mounted.current) return;
    update(next); setResult(next.last_result); setRecognition(null);
    setMessage({ key: value === 'clear' ? 'cleared' : 'edited' });
  });

  const translate = () => run('translating', async () => {
    if (!active.current) return;
    const id = active.current.id;
    try {
      const translated = await api.translate(id);
      if (!mounted.current) return;
      setResult(translated);
      // One authoritative refresh after completion; no polling or jobs.
      update(await api.get(id));
      if (mounted.current) setMessage({ key: translated.error ? 'visualFailed' : 'translated' });
    } catch (cause) {
      if (mounted.current) {
        try { update(await api.get(id)); } catch { /* Preserve visible input for retry. */ }
      }
      throw cause;
    }
  });

  const close = () => run('session', async () => {
    if (!active.current) return;
    try { await api.delete(active.current.id); }
    catch (cause) { if (!(cause instanceof ApiError && cause.status === 404)) throw cause; }
    update(null);
    if (mounted.current) { setResult(null); setRecognition(null); setMessage({ key: 'closed' }); }
  });

  return { language, session, recognition, result, phase, busy: phase !== 'idle', message,
    error: error ? errorMessage(error) : null, rawError: error, offline: error instanceof ApiError && error.status === 0,
    clearConnectionError: () => setError((previous: unknown) => previous instanceof ApiError && previous.status === 0 ? null : previous),
    start, capture, command, translate, close };
}
