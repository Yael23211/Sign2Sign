import { vi, describe, it, expect } from 'vitest';
import { api, ApiError, errorMessage } from './sign2signApi';
import { response, session, translation } from '../test/fixtures';

describe('HTTP contracts', () => {
  it('uses relative endpoints and exact JSON fields', async () => {
    const fetcher = vi.fn().mockImplementation(async () => response(session)); vi.stubGlobal('fetch', fetcher);
    await api.create('ASL');
    expect(fetcher).toHaveBeenCalledWith('/api/v1/sessions', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ source_language: 'ASL' }),
    }));
    await api.command('test-session', 'backspace');
    expect(fetcher).toHaveBeenLastCalledWith('/api/v1/sessions/test-session/commands',
      expect.objectContaining({ body: JSON.stringify({ command: 'backspace' }) }));
  });
  it('sends a Blob unchanged as bytes with its MIME type', async () => {
    const fetcher = vi.fn().mockResolvedValue(response({ session, recognition: null })); vi.stubGlobal('fetch', fetcher);
    const blob = new Blob(['image'], { type: 'image/jpeg' });
    await api.capture(session.id, blob);
    expect(fetcher).toHaveBeenCalledWith('/api/v1/sessions/test-session/captures', {
      method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob,
    });
  });
  it('keeps translation request open and handles DELETE 204', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response(translation)).mockResolvedValueOnce(response(null, 204));
    vi.stubGlobal('fetch', fetcher);
    expect(await api.translate(session.id)).toEqual(translation);
    expect(fetcher).toHaveBeenNthCalledWith(1, '/api/v1/sessions/test-session/translations', { method: 'POST' });
    await expect(api.delete(session.id, true)).resolves.toBeUndefined();
    expect(fetcher).toHaveBeenLastCalledWith('/api/v1/sessions/test-session', { method: 'DELETE', keepalive: true });
  });
  it('normalizes network errors and hides raw backend errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('internal stack')));
    await expect(api.health()).rejects.toMatchObject({ status: 0, code: 'network' });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ error: { code: 'sc04_failed', message: 'SECRET' } }, 502)));
    const error = await api.translate(session.id).catch(cause => cause);
    expect(errorMessage(error)).toContain('Tu texto se conserva');
    expect(errorMessage(error)).not.toContain('SECRET');
  });
  it.each([
    [404, 'session_not_found', 'ya no existe'], [409, 'session_busy', 'ocupada'],
    [413, 'capture_too_large', 'demasiado grande'], [422, 'invalid_image', 'no es válida'],
    [415, 'unsupported_image_type', 'no es válida'], [503, 'session_capacity', 'límite de sesiones'],
    [500, 'translation_failed', 'No se pudo completar'],
  ])('provides friendly error for %s / %s', (status, code, text) => {
    expect(errorMessage(new ApiError(status, code))).toContain(text);
  });
});
