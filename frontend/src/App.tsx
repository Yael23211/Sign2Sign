import { useEffect, useRef, useState } from 'react';
import { api, errorMessage } from './api/sign2signApi';
import { useSign2SignSession } from './hooks/useSign2SignSession';
import { CameraPanel } from './components/CameraPanel';
import { TextBuilderPanel } from './components/TextBuilderPanel';
import { TranslationResult } from './components/TranslationResult';
import { StatusMessage } from './components/StatusMessage';
import type { Language } from './types/api';
import { UiLanguageContext } from './i18n/UiLanguage';
import { translateText, type TextKey, type UiLanguage } from './i18n/messages';
import { ShortcutHelp } from './components/ShortcutHelp';
import { UserGuide } from './components/UserGuide';

export default function App() {
  const flow = useSign2SignSession();
  const [guideOpen, setGuideOpen] = useState(false);
  const guideButton = useRef<HTMLButtonElement>(null);
  const closeGuide = () => { setGuideOpen(false); guideButton.current?.focus(); };
  const [uiOverride, setUiOverride] = useState<UiLanguage | null>(null);
  const locale = uiOverride ?? (flow.language === 'LSM' ? 'es' : 'en');
  const t = (key: TextKey, values?: Record<string, string | number>) => translateText(locale, key, values);
  useEffect(() => {
    document.documentElement.lang = locale;
    document.title = translateText(locale, 'pageTitle');
  }, [locale]);
  const [health, setHealth] = useState<'checking' | 'online' | 'offline'>('checking');
  async function checkHealth() {
    setHealth('checking');
    try {
      const result = await api.health(); setHealth(result.status === 'ok' ? 'online' : 'offline');
      if (result.status === 'ok') flow.clearConnectionError();
    }
    catch { setHealth('offline'); }
  }
  useEffect(() => { void checkHealth(); }, []);
  const offline = health === 'offline' || flow.offline;
  const backendStatus = t(offline ? 'backendOffline' : health === 'checking' ? 'backendChecking' : 'backendOnline');
  return <UiLanguageContext.Provider value={locale}>
    <header className="site-header"><a className="brand" href="/" aria-label={t('home')}><span className="brand-symbol" aria-hidden="true">s²</span>Sign2Sign</a>
      <div className="header-tools">
        <button ref={guideButton} className="guide-access" aria-expanded={guideOpen} aria-controls="user-guide"
          onClick={() => guideOpen ? closeGuide() : setGuideOpen(true)}><span aria-hidden="true">?</span> {t('guideAccess')}</button>
        <div className="ui-language" role="group" aria-label={t('uiLanguage')}>
        <span>{t('uiLanguage')}</span>
        <button lang="es" aria-pressed={locale === 'es'} onClick={() => setUiOverride('es')}>ES</button>
        <span aria-hidden="true">|</span><button lang="en" aria-pressed={locale === 'en'} onClick={() => setUiOverride('en')}>EN</button>
        <button className="auto-language" aria-label={t('useAutomatic')} aria-pressed={uiOverride === null}
          onClick={() => setUiOverride(null)}>{t('automatic')}</button>
      </div>
      <div className="backend-state" role="status" aria-live="polite">
        <button className="backend-indicator" onClick={() => void checkHealth()} disabled={health === 'checking'}
          aria-label={backendStatus} aria-description={t('checkBackend')} title={`${backendStatus}. ${t('checkBackend')}`}>
          <span className={`connection-dot ${offline ? 'offline' : health === 'online' ? 'online' : 'checking'}`} aria-hidden="true" />
        </button></div></div></header>
    <main>
      {guideOpen && <UserGuide onClose={closeGuide} />}
      <section className="intro"><div><span className="eyebrow">{t('slogan')}</span>
        <h1>{t('hero')}<br /><em>{t('heroEm')}</em></h1>
        <p>{t('intro')}</p></div>
        <div className="language-card"><label htmlFor="source-language">{t('source')}</label>
          <select id="source-language" value={flow.language} disabled={flow.busy}
            onChange={event => void flow.start(event.target.value as Language)}>
            <option value="ASL">{t('asl')}</option><option value="LSM">{t('lsm')}</option></select>
          <div className="language-direction"><span>{flow.language}</span><span aria-hidden="true">→</span><span>{flow.language === 'ASL' ? 'LSM' : 'ASL'}</span></div>
          <div className="session-actions"><button className="text-button" disabled={flow.busy} onClick={() => void flow.start(flow.language)}>
            {t(flow.session ? 'newSession' : 'startSession')}</button>
            {flow.session && <button className="text-button muted" disabled={flow.busy} onClick={() => void flow.close()}>{t('closeSession')}</button>}</div>
        </div></section>
      {offline && <StatusMessage error>{t('backendHelp')}</StatusMessage>}
      {flow.error && <StatusMessage error>{errorMessage(flow.rawError, locale)}</StatusMessage>}
      <ShortcutHelp />
      {flow.message?.key === 'noSign' && <StatusMessage>{t('noSign')}</StatusMessage>}
      <div className="flow-status" role="status" aria-live="polite" aria-busy={flow.busy}>
        {flow.busy && <span className="spinner" aria-hidden="true" />}
        {flow.phase === 'translating' ? t('translatingStatus') :
          flow.phase === 'capturing' ? t('capturingStatus') : flow.busy ? t('updating') :
          flow.message && flow.message.key !== 'noSign' ? t(flow.message.key, flow.message.values) : null}
      </div>
      <div className="workspace"><CameraPanel busy={flow.busy} hasSession={!!flow.session} onCapture={image => void flow.capture(image)} />
        <TextBuilderPanel session={flow.session} recognition={flow.recognition} busy={flow.busy} translating={flow.phase === 'translating'}
          onCommand={command => void flow.command(command)} onTranslate={() => void flow.translate()} /></div>
      <TranslationResult result={flow.result} />
    </main>
    <footer><span>Sign2Sign <span aria-hidden="true">/</span> {t('footer')}</span><span>{t('scope')}</span></footer>
  </UiLanguageContext.Provider>;
}
