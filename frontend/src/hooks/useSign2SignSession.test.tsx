import { act, renderHook, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useSign2SignSession } from './useSign2SignSession';
import { response, session } from '../test/fixtures';

it('deletes a newly created session if its response arrives after unmount', async () => {
  let finish!: (response: Response) => void;
  const fetcher = vi.fn((url: string, init?: RequestInit) => init?.method === 'DELETE'
    ? Promise.resolve(response(null, 204)) : new Promise<Response>(resolve => { finish = resolve; }));
  vi.stubGlobal('fetch', fetcher);
  const { result, unmount } = renderHook(useSign2SignSession);
  let pending!: Promise<void>;
  act(() => { pending = result.current.start('ASL'); });
  unmount();
  await act(async () => { finish(response(session, 201)); await pending; });
  expect(fetcher).toHaveBeenLastCalledWith('/api/v1/sessions/test-session', { method: 'DELETE', keepalive: true });
});

it('does not orphan the current session if deleting it fails during a language change', async () => {
  const fetcher = vi.fn((url: string, init?: RequestInit) => Promise.resolve(
    init?.method === 'DELETE' ? response({ error: { code: 'session_busy' } }, 409) : response(session, 201)));
  vi.stubGlobal('fetch', fetcher);
  const { result } = renderHook(useSign2SignSession);
  await act(async () => { await result.current.start('ASL'); });
  await act(async () => { await result.current.start('LSM'); });
  await waitFor(() => expect(result.current.error).toContain('ocupada'));
  expect(result.current.session?.id).toBe(session.id);
  expect(result.current.language).toBe('ASL');
  expect(fetcher.mock.calls.filter(([url]) => url === '/api/v1/sessions')).toHaveLength(1);
});
