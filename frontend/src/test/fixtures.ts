import type { Session, Translation } from '../types/api';

export const translation: Translation = {
  source_language: 'ASL', target_language: 'LSM', recognized_text: 'HELLO', corrected_text: 'HELLO',
  translated_text: 'HOLA', concepts: ['HELLO', 'WATER', 'HELLO'], unresolved: [],
  resources: ['HELLO', 'WATER', 'HELLO'].map((concept_id, i) => ({
    concept_id, language: 'LSM', url: `/api/v1/resources/LSM/${concept_id}`, position: i + 1, total: 3,
  })), latency_seconds: .25, state: 'ready', error: null,
};
export const session: Session = {
  id: 'test-session', source_language: 'ASL', target_language: 'LSM', text: '', state: 'capturing', last_result: null,
};
export const response = (body: unknown, status = 200) => new Response(status === 204 ? null : JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
});
export const recognition = { letter: 'H', confidence: .87, source_language: 'ASL',
  top_k: [{ label: 'H', probability: .87 }, { label: 'A', probability: .1 }] };
