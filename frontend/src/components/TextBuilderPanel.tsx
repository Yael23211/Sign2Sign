import type { Command, Recognition, Session } from '../types/api';
import { RecognitionPanel } from './RecognitionPanel';
import { useI18n } from '../i18n/UiLanguage';
import { useShortcuts } from '../hooks/useShortcuts';

export function TextBuilderPanel({ session, recognition, busy, translating, onCommand, onTranslate }: {
  session: Session | null; recognition: Recognition | null; busy: boolean; translating: boolean;
  onCommand: (command: Command) => void; onTranslate: () => void;
}) {
  const { t } = useI18n();
  useShortcuts(key => {
    if (!session || busy) return false;
    if (key === ' ') { onCommand('space'); return true; }
    if (key === 'backspace' && session.text) { onCommand('backspace'); return true; }
    if (key === 'escape' && session.text) { onCommand('clear'); return true; }
    if (key === 't' && session.text.trim()) { onTranslate(); return true; }
    return false;
  });
  return <section className="panel builder-panel" aria-labelledby="builder-title">
    <div className="panel-heading"><div><span className="eyebrow">{t('builderStep')}</span><h2 id="builder-title">{t('builderTitle')}</h2></div></div>
    <p className="section-intro">{t('builderIntro')}</p>
    <label className="field-label" htmlFor="built-text">{t('builtText')}</label>
    <textarea id="built-text" value={session?.text ?? ''} readOnly placeholder={t('textPlaceholder')} />
    <div className="editing-actions"><button disabled={!session || busy} aria-keyshortcuts="Space" onClick={() => onCommand('space')}>{t('space')} <kbd aria-hidden="true">Space</kbd></button>
      <button disabled={!session || busy || !session.text} aria-keyshortcuts="Backspace" onClick={() => onCommand('backspace')}>{t('backspace')} <kbd aria-hidden="true">Backspace</kbd></button>
      <button disabled={!session || busy || !session.text} aria-keyshortcuts="Escape" onClick={() => onCommand('clear')}>{t('clear')} <kbd aria-hidden="true">Esc</kbd></button></div>
    <RecognitionPanel recognition={recognition} />
    <div className="translate-area"><p>{t('messageReady')}<br /><span>{t('translateHint')}</span></p>
      <button className="button primary" aria-keyshortcuts="T" disabled={!session?.text.trim() || busy} onClick={onTranslate}>
        {t(translating ? 'translating' : 'translate')} <kbd aria-hidden="true">T</kbd></button></div>
  </section>;
}
