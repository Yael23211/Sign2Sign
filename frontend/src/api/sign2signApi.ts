import type { CaptureResponse, Command, Health, Language, Session, Translation } from '../types/api';
import { translateText, type UiLanguage } from '../i18n/messages';

export class ApiError extends Error {
  constructor(public status: number, public code: string) { super(code); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api/v1${path}`, init); }
  catch { throw new ApiError(0, 'network'); }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(response.status, body?.error?.code ?? 'http_error');
  }
  if (response.status === 204) return undefined as T;
  try { return await response.json() as T; }
  catch { throw new ApiError(0, 'invalid_response'); }
}

const json = (body: object): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});
const sessionPath = (id: string) => `/sessions/${encodeURIComponent(id)}`;
export const api = {
  health: () => request<Health>('/health'),
  create: (source_language: Language) => request<Session>('/sessions', json({ source_language })),
  get: (id: string) => request<Session>(sessionPath(id)),
  capture: (id: string, blob: Blob) => request<CaptureResponse>(`${sessionPath(id)}/captures`, {
    method: 'POST', headers: { 'Content-Type': blob.type }, body: blob,
  }),
  command: (id: string, command: Command) => request<Session>(`${sessionPath(id)}/commands`, json({ command })),
  translate: (id: string) => request<Translation>(`${sessionPath(id)}/translations`, { method: 'POST' }),
  delete: (id: string, keepalive = false) => request<void>(sessionPath(id), { method: 'DELETE', keepalive }),
};

export function errorMessage(error: unknown, locale: UiLanguage = 'es'): string {
  const t = (key: Parameters<typeof translateText>[1]) => translateText(locale, key);
  if (!(error instanceof ApiError)) return t('unexpected');
  if (error.code === 'invalid_response') return t('invalidResponse');
  if (error.status === 0) return t('network');
  if (error.status === 409) return t('sessionBusy');
  if (error.status === 404) return t('sessionMissing');
  if (error.status === 413) return t('imageLarge');
  if (error.status === 415 || error.code === 'invalid_image') return t('invalidImage');
  if (error.code === 'text_limit') return t('textLimit');
  if (error.code === 'empty_text') return t('emptyText');
  if (error.status === 502) return t('sc04Failed');
  if (error.status === 503) return t('sessionCapacity');
  return t('operationFailed');
}
