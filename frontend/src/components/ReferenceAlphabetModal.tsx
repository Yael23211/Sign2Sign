import { useEffect, useRef, useState, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { useI18n } from '../i18n/UiLanguage';
import type { Language } from '../types/api';
import { ReferenceAlphabetGrid } from './ReferenceAlphabetGrid';

function ReferenceAlphabetSelector({ onSelect, firstButton }: {
  onSelect: (language: Language) => void; firstButton: RefObject<HTMLButtonElement | null>;
}) {
  const { t } = useI18n();
  return <div className="reference-language-choices" role="group" aria-label={t('referenceSelectLanguage')}>
    <button ref={firstButton} className="button secondary" onClick={() => onSelect('LSM')}>LSM</button>
    <button className="button secondary" onClick={() => onSelect('ASL')}>ASL</button>
  </div>;
}

export function ReferenceAlphabetModal({ onClose, returnFocusTo }: {
  onClose: () => void; returnFocusTo: RefObject<HTMLButtonElement | null>;
}) {
  const { t } = useI18n();
  const dialog = useRef<HTMLDialogElement>(null);
  const firstChoice = useRef<HTMLButtonElement>(null);
  const back = useRef<HTMLButtonElement>(null);
  const [language, setLanguage] = useState<Language | null>(null);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    const overflow = document.body.style.overflow;
    element.showModal(); // Native modal focus containment and inert background.
    document.body.style.overflow = 'hidden';
    const onKey = (event: KeyboardEvent) => {
      // Preserve native button/Tab behavior, but never send keys to the main UI.
      event.stopPropagation();
      if (event.key === 'Escape') { event.preventDefault(); onClose(); }
      if (event.key === 'Tab') {
        const controls = Array.from(element.querySelectorAll<HTMLElement>('button:not(:disabled), [tabindex="0"]'));
        const first = controls[0];
        const last = controls.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault(); last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault(); first?.focus();
        }
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      element.close();
      document.body.style.overflow = overflow;
      returnFocusTo.current?.focus({ preventScroll: true });
    };
  }, [onClose, returnFocusTo]);

  useEffect(() => {
    (language ? back.current : firstChoice.current)?.focus({ preventScroll: true });
    dialog.current?.querySelector('.reference-alphabet-content')?.scrollTo?.({ top: 0 });
  }, [language]);

  return createPortal(<dialog ref={dialog} className="reference-alphabet-modal" aria-modal="true"
    aria-labelledby="reference-alphabet-title" aria-describedby="reference-alphabet-description"
    onCancel={event => { event.preventDefault(); onClose(); }}>
    <div className="reference-alphabet-header">
      <h2 id="reference-alphabet-title">{t('referenceAlphabet')}{language && <span className="pill">{language}</span>}</h2>
      <button className="reference-alphabet-close" aria-label={t('referenceClose')} title={t('referenceClose')}
        onClick={onClose}><span aria-hidden="true">×</span></button>
    </div>
    <div className="reference-alphabet-content">
      <p id="reference-alphabet-description" className="reference-alphabet-intro">
        {t(language ? 'referenceNote' : 'referenceSelectLanguage')}
      </p>
      {language ? <>
        <div className="reference-alphabet-navigation">
          <button ref={back} className="text-button" onClick={() => setLanguage(null)}>← {t('referenceBack')}</button>
          <div className="ui-language" role="group" aria-label={t('referenceSelectLanguage')}>
            {(['LSM', 'ASL'] as const).map(value => <button key={value} aria-pressed={language === value}
              onClick={() => setLanguage(value)}>{value}</button>)}
          </div>
        </div>
        <ReferenceAlphabetGrid language={language} />
      </> : <ReferenceAlphabetSelector onSelect={setLanguage} firstButton={firstChoice} />}
    </div>
  </dialog>, document.body);
}
