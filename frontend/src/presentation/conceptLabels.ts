import { concepts } from '../../../data/vocabulary/sign2sign_vocabulary.json';
import type { Language } from '../types/api';

// Reuse canonical bridge-language text; do not maintain another vocabulary.
const labels = new Map(concepts.map(concept => [concept.concept_id, {
  ASL: concept.english.canonical,
  LSM: concept.spanish.canonical,
}]));

export function getConceptDisplayLabel(conceptId: string, targetLanguage: Language): string {
  // Unknown future concepts get a neutral label, never an invented translation.
  return labels.get(conceptId)?.[targetLanguage] ?? (targetLanguage === 'LSM' ? 'Seña' : 'Sign');
}
