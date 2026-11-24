import { existsSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, it } from 'vitest';
import { referenceAlphabets } from './referenceAlphabet';

it('maps all 26 letters in each language to the actual public JPG/PNG files, without manifests', () => {
  const root = resolve('public');
  for (const language of ['ASL', 'LSM'] as const) {
    const alphabet = referenceAlphabets[language];
    expect(alphabet.map(entry => entry.letter).join('')).toBe('ABCDEFGHIJKLMNOPQRSTUVWXYZ');
    expect(readdirSync(resolve(root, 'reference-alphabet', language.toLowerCase())).filter(name => /^[A-Z]\.(jpg|png)$/.test(name))).toHaveLength(26);
    for (const entry of alphabet) {
      expect(entry.image).toMatch(/^\/reference-alphabet\/(asl|lsm)\/[A-Z]\.(jpg|png)$/);
      expect(existsSync(resolve(root, entry.image.slice(1))), entry.image).toBe(true);
    }
  }
});

it('keeps the exact provisional and static annotations separate by language', () => {
  expect(referenceAlphabets.ASL.filter(entry => entry.provisional)).toHaveLength(0);
  expect(referenceAlphabets.LSM.filter(entry => entry.provisional).map(entry => entry.letter)).toEqual(['E', 'L', 'Z']);
  expect(referenceAlphabets.ASL.filter(entry => entry.staticApproximation).map(entry => entry.letter)).toEqual(['J', 'Z']);
  expect(referenceAlphabets.LSM.filter(entry => entry.staticApproximation).map(entry => entry.letter)).toEqual(['J', 'K', 'Q', 'X', 'Z']);
  expect(referenceAlphabets.LSM.find(entry => entry.letter === 'A')?.image).toMatch(/A.jpg$/);
  expect(referenceAlphabets.LSM.find(entry => entry.letter === 'E')?.image).toMatch(/E.png$/);
});
