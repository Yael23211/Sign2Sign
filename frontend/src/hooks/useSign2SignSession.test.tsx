import { act, renderHook, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { StrictMode } from 'react';
import { useSign2SignSession } from './useSign2SignSession';
import { response, session } from '../test/fixtures';

it('deletes a newly created session if its response arrives after unmount', async () => {
  let finish!: (response: Response) => void;
  const fetcher = vi.fn((url: string, init?: RequestInit) => init?.method === 'DELETE'
    ? Promise.resolve(response(null, 204)) : new Promise<Response>(resolve => { finish = resolve; }));
  vi.stubGlobal('fetch', fetcher);
  const { unmount } = renderHook(useSign2SignSession);
  unmount();
  await act(async () => { finish(response(session, 201)); });
  expect(fetcher).toHaveBeenLastCalledWith('/api/v1/sessions/test-session', { method: 'DELETE', keepalive: true });
});

it('does not orphan the current session if deleting it fails during a language change', async () => {
  const fetcher = vi.fn((url: string, init?: RequestInit) => Promise.resolve(
    init?.method === 'DELETE' ? response({ error: { code: 'session_busy' } }, 409) : response(session, 201)));
  vi.stubGlobal('fetch', fetcher);
  const { result } = renderHook(useSign2SignSession);
  await waitFor(() => expect(result.current.session?.id).toBe(session.id));
  await act(async () => { await result.current.start('LSM'); });
  await waitFor(() => expect(result.current.error).toContain('ocupada'));
  expect(result.current.session?.id).toBe(session.id);
  expect(result.current.language).toBe('ASL');
  expect(fetcher.mock.calls.filter(([url]) => url === '/api/v1/sessions')).toHaveLength(1);
});

it('creates the default ASL session once and respects an explicit close across rerenders', async () => {
  const fetcher = vi.fn((url: string, init?: RequestInit) => Promise.resolve(
    init?.method === 'DELETE' ? response(null, 204) : response(session, 201)));
  vi.stubGlobal('fetch', fetcher);
  const { result, rerender } = renderHook(useSign2SignSession);
  await waitFor(() => expect(result.current.session?.id).toBe(session.id));
  expect(fetcher).toHaveBeenCalledWith('/api/v1/sessions', expect.objectContaining({
    method: 'POST', body: JSON.stringify({ source_language: 'ASL' }),
  }));
  rerender();
  expect(fetcher.mock.calls.filter(([url]) => url === '/api/v1/sessions')).toHaveLength(1);
  await act(async () => { await result.current.close(); });
  rerender();
  expect(result.current.session).toBeNull();
  expect(result.current.busy).toBe(false);
  expect(fetcher.mock.calls.filter(([url]) => url === '/api/v1/sessions')).toHaveLength(1);
});

it('keeps one initial request and one active session through StrictMode effect replay', async () => {
  let finish!: (value: Response) => void;
  const fetcher = vi.fn((url: string, init?: RequestInit) => init?.method === 'DELETE'
    ? Promise.resolve(response(null, 204)) : new Promise<Response>(resolve => { finish = resolve; }));
  vi.stubGlobal('fetch', fetcher);
  const { result, unmount } = renderHook(useSign2SignSession, { wrapper: StrictMode });
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(result.current.busy).toBe(true);
  expect(result.current.session).toBeNull();
  await act(async () => { finish(response(session, 201)); });
  await waitFor(() => expect(result.current.session?.id).toBe(session.id));
  expect(result.current.busy).toBe(false);
  expect(fetcher.mock.calls.filter(([, init]) => init?.method === 'DELETE')).toHaveLength(0);
  unmount();
  expect(fetcher.mock.calls.filter(([, init]) => init?.method === 'DELETE')).toHaveLength(1);
});

it('preserves the initial language after initialization fails and allows the existing start action to retry', async () => {
  const fetcher = vi.fn().mockResolvedValueOnce(response({ error: { code: 'session_capacity' } }, 503))
    .mockImplementation(async () => response(session, 201));
  vi.stubGlobal('fetch', fetcher);
  const { result } = renderHook(useSign2SignSession);
  await waitFor(() => expect(result.current.error).toContain('límite de sesiones'));
  expect(result.current.session).toBeNull();
  expect(result.current.language).toBe('ASL');
  expect(result.current.busy).toBe(false);
  await act(async () => { await result.current.start('ASL'); });
  expect(result.current.session?.id).toBe(session.id);
  expect(result.current.error).toBeNull();
  expect(fetcher).toHaveBeenCalledTimes(2);
});
