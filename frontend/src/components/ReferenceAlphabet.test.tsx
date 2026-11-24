import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { ReferenceAlphabetButton } from './ReferenceAlphabetButton';
import { ReferenceLetterCard } from './ReferenceAlphabetGrid';
import { UiLanguageContext } from '../i18n/UiLanguage';
import { useShortcuts } from '../hooks/useShortcuts';
import { recognition, response, session, translation } from '../test/fixtures';
import App from '../App';

beforeEach(() => {
  // jsdom has no native dialog implementation. Browser validation covers modal behavior.
  Object.defineProperty(HTMLDialogElement.prototype, 'showModal', { configurable: true,
    value: function (this: HTMLDialogElement) { this.setAttribute('open', ''); } });
  Object.defineProperty(HTMLDialogElement.prototype, 'close', { configurable: true,
    value: function (this: HTMLDialogElement) { this.removeAttribute('open'); } });
});

it('opens selection first, shows both complete alphabets, switches and returns without HTTP', async () => {
  const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
  render(<ReferenceAlphabetButton />);
  await userEvent.click(screen.getByRole('button', { name: 'Alfabeto de referencia' }));
  expect(screen.getByRole('dialog', { name: 'Alfabeto de referencia' })).toBeInTheDocument();
  expect(screen.getByText('Selecciona la lengua de señas:')).toBeInTheDocument();
  expect(screen.queryAllByRole('img')).toHaveLength(0);
  await userEvent.click(screen.getByRole('button', { name: 'LSM' }));
  expect(screen.getAllByRole('img')).toHaveLength(26);
  expect(screen.getAllByRole('heading', { level: 3 }).map(heading => heading.textContent).join('')).toBe('ABCDEFGHIJKLMNOPQRSTUVWXYZ');
  expect(screen.getByRole('img', { name: 'LSM, letra A: configuración de referencia' })).toHaveAttribute('src', '/reference-alphabet/lsm/A.jpg');
  expect(screen.getAllByText('Referencia provisional')).toHaveLength(3);
  expect(screen.getAllByText('Representación estática')).toHaveLength(5);
  const zCard = within(screen.getByRole('listitem', { name: 'Z' }));
  expect(zCard.getByText('Referencia provisional')).toBeInTheDocument();
  expect(zCard.getByText('Representación estática')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'ASL' }));
  expect(screen.getAllByRole('img')).toHaveLength(26);
  expect(screen.getByRole('img', { name: 'ASL, letra A: configuración de referencia' })).toHaveAttribute('src', '/reference-alphabet/asl/A.png');
  expect(screen.queryByText('Referencia provisional')).not.toBeInTheDocument();
  expect(screen.getAllByText('Representación estática')).toHaveLength(2);
  await userEvent.click(screen.getByRole('button', { name: /Volver a la selección/ }));
  expect(screen.queryAllByRole('img')).toHaveLength(0);
  expect(screen.getByRole('button', { name: 'LSM' })).toHaveFocus();
  await userEvent.click(screen.getByRole('button', { name: 'Cerrar alfabeto de referencia' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Alfabeto de referencia' })).toHaveFocus();
  expect(fetcher).not.toHaveBeenCalled();
});

it('uses the existing ES/EN context without resetting the selected alphabet', async () => {
  const { rerender } = render(<UiLanguageContext.Provider value="es"><ReferenceAlphabetButton /></UiLanguageContext.Provider>);
  await userEvent.click(screen.getByRole('button', { name: 'Alfabeto de referencia' }));
  await userEvent.click(screen.getByRole('button', { name: 'LSM' }));
  const grid = screen.getByRole('list', { name: 'Alfabeto de referencia LSM' });
  rerender(<UiLanguageContext.Provider value="en"><ReferenceAlphabetButton /></UiLanguageContext.Provider>);
  expect(screen.getByRole('list', { name: 'LSM reference alphabet' })).toBe(grid);
  expect(screen.getByText('The images show capture references corresponding to the configurations used by the Sign2Sign classifiers.')).toBeInTheDocument();
  expect(screen.getAllByText('Provisional reference')).toHaveLength(3);
  expect(screen.getAllByText('Static representation')).toHaveLength(5);
  expect(screen.getByRole('button', { name: 'LSM' })).toHaveAttribute('aria-pressed', 'true');
  await userEvent.click(screen.getByRole('button', { name: 'Close reference alphabet' }));
  expect(screen.getByRole('button', { name: 'Reference alphabet' })).toHaveFocus();
});

it('shows nonfatal placeholders for missing images and accessible annotation explanations', () => {
  render(<ul><ReferenceLetterCard entry={{ letter: 'E', image: '/missing.png', provisional: true, staticApproximation: true }} language="LSM" />
    <ReferenceLetterCard entry={{ letter: 'F', image: '/reference-alphabet/lsm/F.jpg' }} language="LSM" /></ul>);
  const image = screen.getByRole('img', { name: 'LSM, letra E: configuración de referencia' });
  fireEvent.error(image);
  expect(screen.getByRole('status')).toHaveTextContent('Referencia no disponible');
  expect(screen.getByRole('img', { name: 'LSM, letra F: configuración de referencia' })).toBeInTheDocument();
  expect(screen.getByText('Referencia provisional')).toHaveAccessibleDescription('Esta clase utiliza provisionalmente muestras procedentes del conjunto ASL.');
  expect(screen.getByText('Representación estática')).toHaveAccessibleDescription('El prototipo utiliza una configuración estática para esta letra y no analiza su trayectoria de movimiento.');
});

it('consumes Escape and other shortcuts without clearing or capturing in the background, and restores scroll', async () => {
  const action = vi.fn();
  function Background() { useShortcuts(() => { action(); return true; }); return <button>Background</button>; }
  const previous = document.body.style.overflow;
  document.body.style.overflow = 'auto';
  const { unmount } = render(<><Background /><ReferenceAlphabetButton /></>);
  await userEvent.click(screen.getByRole('button', { name: 'Alfabeto de referencia' }));
  expect(document.body.style.overflow).toBe('hidden');
  const close = screen.getByRole('button', { name: 'Cerrar alfabeto de referencia' });
  const last = screen.getByRole('button', { name: 'ASL' });
  close.focus();
  fireEvent.keyDown(close, { key: 'Tab', shiftKey: true });
  expect(last).toHaveFocus();
  fireEvent.keyDown(last, { key: 'Tab' });
  expect(close).toHaveFocus();
  for (const key of ['Enter', ' ', 'Backspace', 't', 'p']) fireEvent.keyDown(screen.getByRole('dialog'), { key });
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
  expect(action).not.toHaveBeenCalled();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(document.body.style.overflow).toBe('auto');
  await userEvent.click(screen.getByRole('button', { name: 'Alfabeto de referencia' }));
  unmount(); expect(document.body.style.overflow).toBe('auto');
  document.body.style.overflow = previous;
});

it('keeps an active session, camera, input and output video intact when opening, navigating and closing', async () => {
  let current = { ...session };
  const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/health')) return response({ status: 'ok', storage: 'memory', workers: 1 });
    if (url === '/api/v1/sessions') return response(current, 201);
    if (init?.method === 'DELETE') return response(null, 204);
    if (url.endsWith('/captures')) { current = { ...current, text: 'H' }; return response({ session: current, recognition }); }
    if (url.endsWith('/translations')) { current = { ...current, state: 'ready', last_result: translation }; return response(translation); }
    return response(current);
  });
  vi.stubGlobal('fetch', fetcher);
  const stop = vi.fn(); const track = { stop, onended: null };
  const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [track], getVideoTracks: () => [track] });
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage: vi.fn() } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(callback => callback(new Blob(['image'], { type: 'image/jpeg' })));
  render(<App />);
  await screen.findByRole('button', { name: 'New session' });
  await userEvent.click(screen.getByRole('button', { name: 'Turn camera on' }));
  const preview = screen.getByLabelText('Camera preview, visually mirrored') as HTMLVideoElement;
  fireEvent.loadedData(preview);
  await userEvent.click(screen.getByRole('button', { name: 'Capture sign' }));
  await userEvent.click(screen.getByRole('button', { name: 'Translate message' }));
  await screen.findByText('Translation completed.');
  const video = screen.getByLabelText('Video of HOLA');
  const stream = preview.srcObject;
  const before = fetcher.mock.calls.length;
  const state = current;
  await userEvent.click(screen.getByRole('button', { name: 'Reference alphabet' }));
  await userEvent.click(screen.getByRole('button', { name: 'LSM' }));
  await userEvent.click(screen.getByRole('button', { name: 'ASL' }));
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
  expect(screen.getByLabelText('Text in progress')).toHaveValue('H');
  expect(screen.getByLabelText('Source sign language')).toHaveValue('ASL');
  expect(screen.getByLabelText('Camera preview, visually mirrored')).toBe(preview);
  expect(preview.srcObject).toBe(stream);
  expect(screen.getByLabelText('Video of HOLA')).toBe(video);
  expect(current).toBe(state);
  expect(stop).not.toHaveBeenCalled();
  expect(getUserMedia).toHaveBeenCalledOnce();
  expect(fetcher).toHaveBeenCalledTimes(before);
});
