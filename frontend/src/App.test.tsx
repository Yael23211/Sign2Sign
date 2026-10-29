import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, it, expect, vi, type Mock } from 'vitest';
import App from './App';
import { recognition, response, session, translation } from './test/fixtures';
import type { Session } from './types/api';

let current: Session;
let fetcher: Mock<(url: string, init?: RequestInit) => Promise<Response>>;
beforeEach(() => {
  current = { ...session };
  fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/health')) return response({ status: 'ok', storage: 'memory', workers: 1 });
    if (url === '/api/v1/sessions') {
      const source = JSON.parse(init?.body as string).source_language;
      current = { ...session, source_language: source, target_language: source === 'ASL' ? 'LSM' : 'ASL' };
      return response(current, 201);
    }
    if (init?.method === 'DELETE') return response(null, 204);
    if (url.endsWith('/captures')) { current = { ...current, text: 'H' }; return response({ session: current, recognition }); }
    if (url.endsWith('/commands')) {
      const command = JSON.parse(init?.body as string).command;
      current = { ...current, text: command === 'space' ? 'H ' : command === 'backspace' ? 'H' : '', last_result: null };
      return response(current);
    }
    if (url.endsWith('/translations')) { current = { ...current, state: 'ready', last_result: translation }; return response(translation); }
    return response(current);
  });
  vi.stubGlobal('fetch', fetcher);
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }], getVideoTracks: () => [] }),
  } });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage: vi.fn() } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(callback => callback(new Blob(['image'], { type: 'image/jpeg' })));
});
async function renderSpanish() {
  const view = render(<App />);
  await userEvent.click(screen.getByRole('button', { name: 'ES' }));
  return view;
}

async function readyToCapture() {
  await userEvent.click(screen.getByRole('button', { name: 'Iniciar sesión' }));
  await userEvent.click(screen.getByRole('button', { name: 'Activar cámara' }));
  fireEvent.loadedData(screen.getByLabelText(/Vista previa/));
  await userEvent.click(screen.getByRole('button', { name: /Capturar seña/ }));
  await waitFor(() => expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H'));
}

describe('SC-01 session integration', () => {
  it('renders the initial interface with accessible controls and backend health', async () => {
    await renderSpanish();
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Lo que quieres decir');
    expect(screen.getByLabelText('Lengua de origen')).toHaveValue('ASL');
    expect(screen.getByRole('button', { name: 'Traducir mensaje' })).toBeDisabled();
    expect(await screen.findByRole('button', { name: 'Backend conectado' })).toBeInTheDocument();
  });
  it('deletes the previous session before switching language and cleans up on unmount', async () => {
    const { unmount } = await renderSpanish();
    await userEvent.click(screen.getByRole('button', { name: 'Iniciar sesión' }));
    await userEvent.selectOptions(screen.getByLabelText('Lengua de origen'), 'LSM');
    await waitFor(() => expect(screen.getByLabelText('Lengua de origen')).toHaveValue('LSM'));
    const calls = fetcher.mock.calls;
    const deletion = calls.findIndex(([, init]) => init?.method === 'DELETE');
    let secondCreation = -1;
    calls.forEach(([url], i) => { if (url === '/api/v1/sessions') secondCreation = i; });
    expect(deletion).toBeLessThan(secondCreation);
    expect(JSON.parse(calls[secondCreation][1]!.body as string)).toEqual({ source_language: 'LSM' });
    unmount();
    expect(fetcher).toHaveBeenLastCalledWith('/api/v1/sessions/test-session', { method: 'DELETE', keepalive: true });
  });
  it('uses backend capture and command responses as authoritative text', async () => {
    await renderSpanish(); await readyToCapture();
    expect(screen.getAllByText('87.0 %')).toHaveLength(2);
    await userEvent.click(screen.getByText('Ver alternativas (Top-k)'));
    expect(screen.getByText('10.0 %')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Espacio' }));
    expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H ');
    await userEvent.click(screen.getByRole('button', { name: 'Borrar último' }));
    expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H');
    await userEvent.click(screen.getByRole('button', { name: 'Limpiar' }));
    expect(screen.getByLabelText('Texto en construcción')).toHaveValue('');
  });
  it('renders translation, refreshes once and repeats videos without retranslating', async () => {
    await renderSpanish(); await readyToCapture();
    await userEvent.click(screen.getByRole('button', { name: 'Traducir mensaje' }));
    expect(await screen.findByText('HOLA')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Repetir secuencia' }));
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/translations'))).toHaveLength(1);
    expect(fetcher.mock.calls.filter(([url, init]) => url.endsWith('/test-session') && !init)).toHaveLength(1);
  });
  it('blocks duplicated translation and editing while the request is open', async () => {
    await renderSpanish(); await readyToCapture();
    const original = fetcher.getMockImplementation()!;
    let finish!: (value: Response) => void;
    fetcher.mockImplementation((url, init) => url.endsWith('/translations') ? new Promise(resolve => { finish = resolve; }) : original(url, init));
    const button = screen.getByRole('button', { name: 'Traducir mensaje' });
    fireEvent.click(button); fireEvent.click(button);
    expect(screen.getByRole('button', { name: /Traduciendo/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Espacio' })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Capturar seña/ })).toBeDisabled();
    expect(screen.getByLabelText('Lengua de origen')).toBeDisabled();
    await act(async () => finish(response(translation)));
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/translations'))).toHaveLength(1);
  });
  it('preserves text and allows retry after SC-04 fails', async () => {
    await renderSpanish(); await readyToCapture();
    const original = fetcher.getMockImplementation()!;
    fetcher.mockImplementation((url, init) => url.endsWith('/translations') ? Promise.resolve(response({ error: { code: 'sc04_failed' } }, 502)) : original(url, init));
    await userEvent.click(screen.getByRole('button', { name: 'Traducir mensaje' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Tu texto se conserva');
    expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H');
    expect(screen.getByRole('button', { name: 'Traducir mensaje' })).toBeEnabled();
  });
  it('shows no-hand response without adding text', async () => {
    await renderSpanish(); await readyToCapture();
    const original = fetcher.getMockImplementation()!;
    fetcher.mockImplementation((url, init) => url.endsWith('/captures') ? Promise.resolve(response({ session: current, recognition: null })) : original(url, init));
    await userEvent.click(screen.getByRole('button', { name: /Capturar seña/ }));
    expect(await screen.findByText('No se detectó una seña válida. Intenta nuevamente.')).toBeInTheDocument();
    expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H');
  });
  it('shows backend unavailable without exposing an exception', async () => {
    fetcher.mockRejectedValue(new Error('private stack'));
    await renderSpanish();
    expect(await screen.findByRole('button', { name: 'Backend sin conexión' })).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('Inicia FastAPI');
  });
  it.each([[404, 'session_not_found', 'ya no existe'], [409, 'session_busy', 'ocupada']])(
    'handles session failure %s without inventing a new session', async (status, code, message) => {
      await renderSpanish(); await readyToCapture();
      const original = fetcher.getMockImplementation()!;
      fetcher.mockImplementation((url, init) => url.endsWith('/commands')
        ? Promise.resolve(response({ error: { code } }, status)) : original(url, init));
      await userEvent.click(screen.getByRole('button', { name: 'Espacio' }));
      expect(await screen.findByRole('alert')).toHaveTextContent(message);
      expect(fetcher.mock.calls.filter(([url]) => url === '/api/v1/sessions')).toHaveLength(1);
      if (status === 404) expect(screen.getByRole('button', { name: 'Iniciar sesión' })).toBeEnabled();
      else expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H');
    });
  it('closes the session explicitly and attempts cleanup on pagehide', async () => {
    await renderSpanish();
    await userEvent.click(screen.getByRole('button', { name: 'Iniciar sesión' }));
    fireEvent(window, new Event('pagehide'));
    expect(fetcher).toHaveBeenLastCalledWith('/api/v1/sessions/test-session', { method: 'DELETE', keepalive: true });
    await userEvent.click(screen.getByRole('button', { name: 'Cerrar sesión' }));
    expect(await screen.findByRole('button', { name: 'Iniciar sesión' })).toBeEnabled();
  });
});

function shortcut(key: string) {
  (document.activeElement as HTMLElement)?.blur();
  fireEvent.keyDown(document.body, { key });
}

describe('incremental UX V2', () => {
  it('translates no-detection feedback into a nonfatal warning without changing camera behavior', async () => {
    await renderSpanish(); await readyToCapture();
    const original = fetcher.getMockImplementation()!;
    fetcher.mockImplementation((url, init) => url.endsWith('/captures')
      ? Promise.resolve(response({ session: current, recognition: null })) : original(url, init));
    await userEvent.click(screen.getByRole('button', { name: 'EN' }));
    await userEvent.click(screen.getByRole('button', { name: /Capture sign/ }));
    expect(await screen.findByText('No valid sign was detected. Please try again.')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Text in progress')).toHaveValue('H');
  });
  it('defaults to English for ASL, Spanish for LSM, and follows automatic source changes', async () => {
    render(<App />);
    expect(screen.getByLabelText('Source sign language')).toHaveValue('ASL');
    expect(document.documentElement.lang).toBe('en');
    await userEvent.selectOptions(screen.getByLabelText('Source sign language'), 'LSM');
    expect(await screen.findByLabelText('Lengua de origen')).toHaveValue('LSM');
    expect(document.documentElement.lang).toBe('es');
    await userEvent.selectOptions(screen.getByLabelText('Lengua de origen'), 'ASL');
    expect(await screen.findByLabelText('Source sign language')).toHaveValue('ASL');
    expect(document.documentElement.lang).toBe('en');
  });

  it('keeps manual UI override across sessions without changing source or sending requests', async () => {
    render(<App />);
    await userEvent.click(screen.getByRole('button', { name: 'Start session' }));
    const before = fetcher.mock.calls.length;
    await userEvent.click(screen.getByRole('button', { name: 'ES' }));
    expect(screen.getByLabelText('Lengua de origen')).toHaveValue('ASL');
    await userEvent.click(screen.getByRole('button', { name: 'EN' }));
    expect(screen.getByLabelText('Source sign language')).toHaveValue('ASL');
    expect(fetcher).toHaveBeenCalledTimes(before);
    await userEvent.click(screen.getByRole('button', { name: 'ES' }));
    await userEvent.selectOptions(screen.getByLabelText('Lengua de origen'), 'LSM');
    await waitFor(() => expect(screen.getByLabelText('Lengua de origen')).toHaveValue('LSM'));
    await userEvent.selectOptions(screen.getByLabelText('Lengua de origen'), 'ASL');
    await waitFor(() => expect(screen.getByLabelText('Lengua de origen')).toHaveValue('ASL'));
    await userEvent.click(screen.getByRole('button', { name: 'Usar idioma automático' }));
    expect(screen.getByLabelText('Source sign language')).toHaveValue('ASL');
  });

  it('captures with Enter only while the camera is active', async () => {
    await renderSpanish();
    await userEvent.click(screen.getByRole('button', { name: 'Iniciar sesión' }));
    shortcut('Enter'); expect(fetcher.mock.calls.some(([url]) => url.endsWith('/captures'))).toBe(false);
    await userEvent.click(screen.getByRole('button', { name: 'Activar cámara' }));
    fireEvent.loadedData(screen.getByLabelText(/Vista previa/));
    shortcut('Enter');
    await waitFor(() => expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H'));
    expect(screen.getByLabelText('Captura confirmada')).toHaveTextContent('✓');
    await userEvent.click(screen.getByRole('button', { name: 'Desactivar cámara' }));
    shortcut('Enter');
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/captures'))).toHaveLength(1);
  });

  it.each([[' ', 'space'], ['Backspace', 'backspace'], ['Escape', 'clear']])(
    '%s sends the existing %s command', async (key, command) => {
      await renderSpanish(); await readyToCapture(); shortcut(key);
      await waitFor(() => expect(fetcher).toHaveBeenCalledWith('/api/v1/sessions/test-session/commands',
        expect.objectContaining({ body: JSON.stringify({ command }) })));
    });

  it('T translates once while busy, then P repeats locally and scrolls only on a new result', async () => {
    const scroll = vi.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: scroll });
    await renderSpanish(); await readyToCapture();
    expect(scroll).not.toHaveBeenCalled();
    const original = fetcher.getMockImplementation()!;
    let finish!: (value: Response) => void;
    fetcher.mockImplementation((url, init) => url.endsWith('/translations') ? new Promise(resolve => { finish = resolve; }) : original(url, init));
    shortcut('t'); shortcut('t'); shortcut('Enter'); shortcut(' ');
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/translations'))).toHaveLength(1);
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/captures'))).toHaveLength(1);
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/commands'))).toHaveLength(0);
    await act(async () => finish(response(translation)));
    expect(scroll).toHaveBeenCalledOnce();
    expect(scroll).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });
    fireEvent.ended(screen.getByLabelText('Video de HELLO'));
    expect(screen.getByText('2 / 3')).toBeInTheDocument();
    shortcut('p'); expect(screen.getByText('1 / 3')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'EN' }));
    expect(scroll).toHaveBeenCalledOnce();
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/translations'))).toHaveLength(1);
  });

  it('does not send repeated captures before the first response arrives', async () => {
    await renderSpanish(); await readyToCapture();
    const original = fetcher.getMockImplementation()!;
    let finish!: (value: Response) => void;
    fetcher.mockImplementation((url, init) => url.endsWith('/captures') ? new Promise(resolve => { finish = resolve; }) : original(url, init));
    shortcut('Enter'); shortcut('Enter');
    await waitFor(() => expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/captures'))).toHaveLength(2));
    await act(async () => finish(response({ session: current, recognition })));
  });

  it.each(['Capturar seña', 'Espacio', 'Borrar último', 'Limpiar', 'Nueva sesión'])(
    'clears the translation-completed status on %s', async button => {
      await renderSpanish(); await readyToCapture();
      await userEvent.click(screen.getByRole('button', { name: 'Traducir mensaje' }));
      expect(await screen.findByText('Traducción terminada.')).toBeInTheDocument();
      await userEvent.click(screen.getByRole('button', { name: new RegExp(`^${button}`) }));
      expect(screen.queryByText('Traducción terminada.')).not.toBeInTheDocument();
    });

  it('does nothing for P without a sequence and reveals lightweight shortcut help', async () => {
    await renderSpanish(); const before = fetcher.mock.calls.length;
    shortcut('p'); expect(fetcher).toHaveBeenCalledTimes(before);
    await userEvent.click(screen.getByText(/⌨ Atajos/));
    expect(screen.getByText('Separar palabras')).toBeVisible();
  });
});

describe('user orientation panel', () => {
  it('opens in English and updates the full guide with the existing ES/EN selector', async () => {
    render(<App />);
    await screen.findByRole('button', { name: 'Backend connected' });
    const requests = fetcher.mock.calls.length;
    await userEvent.click(screen.getByRole('button', { name: 'How to use' }));
    const guide = screen.getByRole('region', { name: 'How to use Sign2Sign' });
    expect(within(guide).queryByText('Select Start session to begin.')).not.toBeInTheDocument();
    expect(within(guide).getByRole('heading', { name: 'How to use Sign2Sign' })).toHaveFocus();
    for (const title of ['Select your language', 'Turn on the camera', 'Build your message', 'Translate', 'View the result', 'About Sign2Sign', 'Prototype scope']) {
      expect(within(guide).getByRole('heading', { name: title })).toBeInTheDocument();
    }
    expect(within(guide).getByText('Does not perform continuous sign language recognition.')).toBeInTheDocument();
    expect(within(guide).getByText(/hearing person/)).toBeInTheDocument();
    expect(within(guide).getByText('Enter')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'ES' }));
    const spanishGuide = screen.getByRole('region', { name: 'Cómo usar Sign2Sign' });
    expect(within(spanishGuide).queryByText('Selecciona Iniciar sesión para comenzar.')).not.toBeInTheDocument();
    for (const title of ['Selecciona tu lengua', 'Activa la cámara', 'Construye tu mensaje', 'Traduce', 'Consulta el resultado', 'Acerca de Sign2Sign', 'Alcance del prototipo']) {
      expect(within(spanishGuide).getByRole('heading', { name: title })).toBeInTheDocument();
    }
    expect(within(spanishGuide).getByText('No realiza reconocimiento continuo de lengua de señas.')).toBeInTheDocument();
    expect(within(spanishGuide).getByText(/persona oyente/)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(requests);
    await userEvent.click(screen.getByRole('button', { name: 'Volver a la traducción' }));
    expect(screen.queryByRole('region', { name: 'Cómo usar Sign2Sign' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cómo usar' })).toHaveFocus();
    expect(fetcher).toHaveBeenCalledTimes(requests);
  });
  it('opens and closes in Spanish without touching the active session, camera or message', async () => {
    await renderSpanish(); await readyToCapture();
    const preview = screen.getByLabelText(/Vista previa/) as HTMLVideoElement;
    const stream = preview.srcObject;
    const requests = fetcher.mock.calls.length;
    const sessionBefore = { ...current };
    const cameraCalls = vi.mocked(navigator.mediaDevices.getUserMedia).mock.calls.length;
    await userEvent.click(screen.getByRole('button', { name: 'Cómo usar' }));
    expect(screen.getByRole('region', { name: 'Cómo usar Sign2Sign' })).toBeVisible();
    expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H');
    expect(screen.getByLabelText(/Vista previa/)).toBe(preview);
    await userEvent.click(screen.getByRole('button', { name: 'Cerrar guía' }));
    expect(screen.getByRole('button', { name: 'Cómo usar' })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByRole('button', { name: 'Cómo usar' })).toHaveFocus();
    expect(screen.getByLabelText('Lengua de origen')).toHaveValue('ASL');
    expect(preview.srcObject).toBe(stream);
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(cameraCalls);
    expect(current).toEqual(sessionBefore);
    expect(fetcher).toHaveBeenCalledTimes(requests);
    await userEvent.click(screen.getByRole('button', { name: 'Espacio' }));
    expect(fetcher).toHaveBeenLastCalledWith('/api/v1/sessions/test-session/commands',
      expect.objectContaining({ body: JSON.stringify({ command: 'space' }) }));
    expect(screen.getByLabelText('Texto en construcción')).toHaveValue('H ');
  });
  it('keeps the current output video mounted and allows closing from the header', async () => {
    await renderSpanish(); await readyToCapture();
    await userEvent.click(screen.getByRole('button', { name: 'Traducir mensaje' }));
    await screen.findByText('Traducción terminada.');
    const video = screen.getByLabelText('Video de HELLO');
    const requests = fetcher.mock.calls.length;
    await userEvent.click(screen.getByRole('button', { name: 'Cómo usar' }));
    await userEvent.click(screen.getByRole('button', { name: 'Cómo usar' }));
    expect(screen.queryByRole('region', { name: 'Cómo usar Sign2Sign' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Video de HELLO')).toBe(video);
    expect(fetcher).toHaveBeenCalledTimes(requests);
  });
});

describe('small visual refinements', () => {
  it.each([
    ['ES', true, 'Backend conectado'], ['ES', false, 'Backend sin conexión'],
    ['EN', true, 'Backend connected'], ['EN', false, 'Backend offline'],
  ] as const)('keeps only an accessible indicator for %s, connected=%s', async (language, connected, label) => {
    if (!connected) fetcher.mockRejectedValue(new Error('offline'));
    render(<App />);
    if (language === 'ES') await userEvent.click(screen.getByRole('button', { name: 'ES' }));
    const indicator = await screen.findByRole('button', { name: label });
    expect(indicator).toHaveAttribute('aria-label', label);
    expect(indicator).toHaveAttribute('title', expect.stringContaining(label));
    expect(indicator).toHaveTextContent('');
    expect(indicator.parentElement?.textContent).toBe('');
    expect(screen.queryByText(label)).not.toBeInTheDocument();
    const dot = indicator.querySelector('.connection-dot');
    expect(dot).toBeInTheDocument();
    expect(dot).toHaveClass(connected ? 'online' : 'offline');
    expect(dot).toHaveAttribute('aria-hidden', 'true');
  });

  it('preserves checking state and manual health recheck without a visible refresh icon', async () => {
    let finish!: (value: Response) => void;
    fetcher.mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    render(<App />);
    const checking = screen.getByRole('button', { name: 'Checking connection…' });
    expect(checking).toBeDisabled();
    expect(checking.querySelector('.connection-dot')).toHaveClass('checking');
    expect(checking.textContent).toBe('');
    await act(async () => finish(response({ status: 'ok', storage: 'memory', workers: 1 })));
    await userEvent.click(screen.getByRole('button', { name: 'Backend connected' }));
    expect(screen.getByRole('button', { name: 'Checking connection…' })).toBeDisabled();
    expect(fetcher).toHaveBeenCalledTimes(2);
    await act(async () => finish(response({ status: 'offline' })));
    expect(screen.getByRole('button', { name: 'Backend offline' })).toBeEnabled();
  });

  it.each(['ES', 'EN'] as const)('shows compact button key hints in %s without changing action names', async language => {
    await renderSpanish(); await readyToCapture();
    await userEvent.click(screen.getByRole('button', { name: 'Traducir mensaje' }));
    await screen.findByText('Traducción terminada.');
    if (language === 'EN') await userEvent.click(screen.getByRole('button', { name: 'EN' }));
    const controls = language === 'ES'
      ? [['Capturar seña', 'Enter'], ['Espacio', 'Space'], ['Borrar último', 'Backspace'], ['Limpiar', 'Esc'], ['Traducir mensaje', 'T'], ['Repetir secuencia', 'P']]
      : [['Capture sign', 'Enter'], ['Space', 'Space'], ['Delete last', 'Backspace'], ['Clear', 'Esc'], ['Translate message', 'T'], ['Repeat sequence', 'P']];
    for (const [name, key] of controls) {
      const button = screen.getByRole('button', { name });
      const hint = button.querySelector('kbd');
      expect(hint).toHaveTextContent(key);
      expect(hint).toHaveAttribute('aria-hidden', 'true');
      expect(button).toHaveAttribute('aria-keyshortcuts', key === 'Esc' ? 'Escape' : key);
    }
  });
});
