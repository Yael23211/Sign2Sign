import { useId, useState } from 'react';
import { referenceAlphabets, type ReferenceLetter } from '../data/referenceAlphabet';
import { useI18n } from '../i18n/UiLanguage';
import type { Language } from '../types/api';

export function ReferenceLetterCard({ entry, language }: { entry: ReferenceLetter; language: Language }) {
  const { t } = useI18n();
  const [unavailable, setUnavailable] = useState(false);
  const id = useId();
  return <li className="reference-letter-card" aria-labelledby={`${id}-letter`}>
    <h3 id={`${id}-letter`}>{entry.letter}</h3>
    <div className="reference-letter-image">
      {unavailable ? <p role="status">{t('referenceUnavailable')}</p> :
        <img src={entry.image} loading="lazy" decoding="async" onError={() => setUnavailable(true)}
          alt={t('referenceImageAlt', { language, letter: entry.letter })} />}
    </div>
    <div className="reference-letter-badges">
      {entry.provisional && <span className="reference-badge" tabIndex={0} aria-describedby={`${id}-provisional`}>
        {t('referenceProvisional')}<span id={`${id}-provisional`} className="reference-tooltip" role="tooltip">{t('referenceProvisionalNote')}</span>
      </span>}
      {entry.staticApproximation && <span className="reference-badge" tabIndex={0} aria-describedby={`${id}-static`}>
        {t('referenceStatic')}<span id={`${id}-static`} className="reference-tooltip" role="tooltip">{t('referenceStaticNote')}</span>
      </span>}
    </div>
  </li>;
}

export function ReferenceAlphabetGrid({ language }: { language: Language }) {
  const { t } = useI18n();
  return <ul className="reference-alphabet-grid" aria-label={t('referenceGrid', { language })}>
    {referenceAlphabets[language].map(entry => <ReferenceLetterCard key={`${language}-${entry.letter}`} entry={entry} language={language} />)}
  </ul>;
}
