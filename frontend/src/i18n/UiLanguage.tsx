import { createContext, useContext } from 'react';
import { translateText, type TextKey, type UiLanguage } from './messages';

export const UiLanguageContext = createContext<UiLanguage>('es');
export function useI18n() {
  const locale = useContext(UiLanguageContext);
  return { locale, t: (key: TextKey, values?: Record<string, string | number>) => translateText(locale, key, values) };
}
