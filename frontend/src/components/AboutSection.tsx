import { useI18n } from '../i18n/UiLanguage';

export function AboutSection() {
  const { t } = useI18n();
  return <section className="about-section" aria-labelledby="about-title">
    <div className="about-columns">
      <div><h2 id="about-title">{t('guideAbout')}</h2>
        <p>{t('guideAboutText')}</p><p>{t('guideAboutProcess')}</p></div>
      <section className="about-scope" aria-labelledby="about-scope-title"><h3 id="about-scope-title">{t('guideScope')}</h3>
        <ul>{(['guideScopeSpelling', 'guideScopeContinuous', 'guideScopeVocabulary', 'guideScopeGrammar'] as const)
          .map(key => <li key={key}>{t(key)}</li>)}</ul></section>
    </div>
    <p className="about-academic">{t('guideAcademic')}</p>
  </section>;
}
