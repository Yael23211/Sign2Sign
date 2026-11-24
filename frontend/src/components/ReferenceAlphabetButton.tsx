import { useCallback, useRef, useState } from 'react';
import { useI18n } from '../i18n/UiLanguage';
import { ReferenceAlphabetModal } from './ReferenceAlphabetModal';

export function ReferenceAlphabetButton() {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const close = useCallback(() => setOpen(false), []);
  return <>
    <button ref={button} className="guide-access" aria-haspopup="dialog" aria-expanded={open}
      onClick={() => setOpen(true)}>{t('referenceAlphabet')}</button>
    {open && <ReferenceAlphabetModal onClose={close} returnFocusTo={button} />}
  </>;
}
