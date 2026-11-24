import { describe, expect, it } from 'vitest';
import { concepts } from '../../../data/vocabulary/sign2sign_vocabulary.json';
import { getConceptDisplayLabel } from './conceptLabels';

describe('canonical display labels', () => {
  it('reuses both canonical labels for every existing concept', () => {
    for (const concept of concepts) {
      expect(getConceptDisplayLabel(concept.concept_id, 'LSM')).toBe(concept.spanish.canonical);
      expect(getConceptDisplayLabel(concept.concept_id, 'ASL')).toBe(concept.english.canonical);
    }
  });
  it('does not expose grammatical identifiers or invent a translation for unknown concepts', () => {
    expect(getConceptDisplayLabel('FIRST_PERSON_SINGULAR', 'LSM')).toBe('YO');
    expect(getConceptDisplayLabel('FIRST_PERSON_SINGULAR', 'ASL')).toBe('I');
    expect(getConceptDisplayLabel('HOW_ARE_YOU', 'ASL')).toBe('HOW ARE YOU');
    expect(getConceptDisplayLabel('FUTURE_TECHNICAL_ID', 'LSM')).toBe('Seña');
    expect(getConceptDisplayLabel('FUTURE_TECHNICAL_ID', 'ASL')).toBe('Sign');
  });
});
