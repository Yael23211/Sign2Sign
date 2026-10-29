import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { CameraPanel } from './CameraPanel';

describe('deliberate camera capture', () => {
  it('only takes a frame on click, does not flip canvas pixels, and releases tracks', async () => {
    const user = userEvent.setup(); const stop = vi.fn();
    const track = { stop, onended: null };
    const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [track], getVideoTracks: () => [track] });
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } });
    const drawImage = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(callback => callback(new Blob(['jpeg'], { type: 'image/jpeg' })));
    const capture = vi.fn();
    const { unmount } = render(<CameraPanel busy={false} hasSession onCapture={image => { void image().then(capture); }} />);
    await user.click(screen.getByRole('button', { name: 'Activar cámara' }));
    fireEvent.loadedData(screen.getByLabelText(/Vista previa/));
    expect(getUserMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
    expect(drawImage).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: /Capturar seña/ }));
    await waitFor(() => expect(capture).toHaveBeenCalledOnce());
    expect(drawImage).toHaveBeenCalledWith(screen.getByLabelText(/Vista previa/), 0, 0, 1280, 720);
    unmount(); expect(stop).toHaveBeenCalled();
  });
  it.each([['NotAllowedError', 'Permiso de cámara denegado'], ['NotFoundError', 'No se encontró una cámara'], ['NotReadableError', 'cámara está ocupada']])(
    'shows an understandable camera error for %s', async (name, message) => {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true,
        value: { getUserMedia: vi.fn().mockRejectedValue(new DOMException('private', name)) } });
      render(<CameraPanel busy={false} hasSession onCapture={vi.fn()} />);
      await userEvent.click(screen.getByRole('button', { name: 'Activar cámara' }));
      expect(await screen.findByRole('alert')).toHaveTextContent(message);
    });
  it('stops a stream that arrives after unmount', async () => {
    const stop = vi.fn(); let resolve!: (stream: unknown) => void;
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true,
      value: { getUserMedia: () => new Promise(value => { resolve = value; }) } });
    const { unmount } = render(<CameraPanel busy={false} hasSession onCapture={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Activar cámara' }));
    unmount();
    await act(async () => resolve({ getTracks: () => [{ stop }] }));
    expect(stop).toHaveBeenCalledOnce();
  });
});
