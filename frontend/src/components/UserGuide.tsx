import { useEffect, useRef } from 'react';
import { useI18n } from '../i18n/UiLanguage';
import type { TextKey } from '../i18n/messages';

const steps: { title: TextKey; text: TextKey; icon: string }[] = [
  { title: 'guideLanguageTitle', text: 'guideLanguageText', icon: 'ASL ↔ LSM' },
  { title: 'guideCameraTitle', text: 'guideCameraText', icon: '◉' },
  { title: 'guideBuildTitle', text: 'guideBuildText', icon: 'Aa' },
  { title: 'guideTranslateTitle', text: 'guideTranslateText', icon: '→' },
  { title: 'guideResultTitle', text: 'guideResultText', icon: '▷' },
];

export function UserGuide({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const title = useRef<HTMLHeadingElement>(null);
  useEffect(() => { title.current?.focus(); }, []);
  return <section id="user-guide" className="panel user-guide" aria-labelledby="guide-title">
    <div className="panel-heading"><div>
      <h2 id="guide-title" ref={title} tabIndex={-1}>{t('guideTitle')}</h2>
      <p className="guide-intro">{t('guideIntro')}</p>
    </div><button onClick={onClose}>{t('guideClose')}</button></div>
    <ol className="guide-steps" role="list">
      {steps.map((step, index) => <li className="guide-step" key={step.title}>
        <div className="guide-step-heading"><span className="guide-number" aria-hidden="true">{index + 1}</span>
          <span className="guide-icon" aria-hidden="true">{step.icon}</span></div>
        <h3>{t(step.title)}</h3><p>{t(step.text)}</p>
        {index === 2 && <><dl className="guide-keys">
          {[['Enter', t('shortcutCapture')], ['Space', t('space')], ['Backspace', t('backspace')], ['Escape', t('clear')]]
            .map(([key, label]) => <div key={key}><dt><kbd>{key}</kbd></dt><dd>{label}</dd></div>)}
        </dl><p className="guide-note">{t('shortcutsHint')}</p></>}
        {index === 3 && <p className="guide-note"><kbd>T</kbd> → {t('shortcutTranslate')}</p>}
        {index === 4 && <><p>{t('guideVideos')}</p><p className="guide-note">{t('guideTools')}</p>
          <p className="guide-note"><kbd>P</kbd> → {t('repeat')}</p></>}
      </li>)}
    </ol>
    <p className="guide-usage-note">{t('guideSpeech')}</p>
    <div className="guide-return"><button className="button primary" onClick={onClose}>{t('guideReturn')}</button></div>
  </section>;
}
