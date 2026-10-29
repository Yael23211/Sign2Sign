import { useI18n } from '../i18n/UiLanguage';

export function ShortcutHelp() {
  const { t } = useI18n();
  return <details className="shortcut-help"><summary>⌨ {t('shortcuts')}</summary>
    <p>{t('shortcutsHint')}</p><dl>
      {[['Enter', t('shortcutCapture')], [t('spaceKey'), t('shortcutSpace')], ['Backspace', t('backspace')],
        ['Esc', t('clear')], ['T', t('shortcutTranslate')], ['P', t('repeat')]].map(([key, label]) =>
        <div key={key}><dt><kbd>{key}</kbd></dt><dd>{label}</dd></div>)}
    </dl></details>;
}
