// Mirrors src/api/schemas.py. Domain logic remains in SC-02–SC-07.
export type Language = 'ASL' | 'LSM';
export type SessionState = 'capturing' | 'processing' | 'ready' | 'failed' | 'closed';
export type Command = 'space' | 'backspace' | 'clear';
export interface VisualResource {
  concept_id: string; language: Language; url: string; position: number; total: number;
}
export interface Translation {
  source_language: Language; target_language: Language;
  recognized_text: string; corrected_text: string; translated_text: string;
  concepts: string[]; unresolved: string[]; resources: VisualResource[];
  latency_seconds: number; state: SessionState;
  error: { stage: 'SC-04' | 'SC-05' | 'SC-06'; code: string; message: string } | null;
}
export interface Session {
  id: string; source_language: Language; target_language: Language;
  state: SessionState; text: string; last_result: Translation | null;
}
export interface Recognition {
  source_language?: Language | null; letter: string; confidence: number;
  top_k: { label: string; probability: number }[]; detection_variant?: string | null;
}
export interface CaptureResponse { session: Session; recognition: Recognition | null }
export interface Health { status: string; storage: string; workers: number }
