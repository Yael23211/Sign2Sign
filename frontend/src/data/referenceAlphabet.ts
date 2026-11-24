import type { Language } from '../types/api';

export type ReferenceLetter = {
  letter: string;
  image: string;
  provisional?: boolean;
  staticApproximation?: boolean;
};

type AlphabetConfig = {
  defaultExtension: 'jpg' | 'png';
  extensionOverrides: Record<string, 'jpg' | 'png'>;
  provisional: readonly string[];
  staticApproximation: readonly string[];
};

// Public filenames inspected directly. Manifests are not read by the UI.
export const referenceAlphabetConfig: Record<Language, AlphabetConfig> = {
  ASL: {
    defaultExtension: 'png', extensionOverrides: {}, provisional: [],
    staticApproximation: ['J', 'Z'],
  },
  LSM: {
    defaultExtension: 'jpg', extensionOverrides: { E: 'png', L: 'png', Z: 'png' }, provisional: ['E', 'L', 'Z'],
    staticApproximation: ['J', 'K', 'Q', 'X', 'Z'],
  },
};

function buildAlphabet(language: Language): ReferenceLetter[] {
  const config = referenceAlphabetConfig[language];
  return Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ', letter => ({
    letter,
    image: `/reference-alphabet/${language.toLowerCase()}/${letter}.${config.extensionOverrides[letter] ?? config.defaultExtension}`,
    provisional: config.provisional.includes(letter),
    staticApproximation: config.staticApproximation.includes(letter),
  }));
}

export const referenceAlphabets: Record<Language, ReferenceLetter[]> = {
  ASL: buildAlphabet('ASL'),
  LSM: buildAlphabet('LSM'),
};
