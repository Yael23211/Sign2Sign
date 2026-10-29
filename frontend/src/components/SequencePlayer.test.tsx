import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { SequencePlayer } from './SequencePlayer';
import { translation } from '../test/fixtures';

describe('ordered sequence playback', () => {
  it('autoplays, advances on ended, stops at end and repeats without HTTP', async () => {
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
    const user = userEvent.setup();
    render(<SequencePlayer resources={translation.resources} />);
    expect(screen.getByText('1 / 3')).toBeInTheDocument();
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
    fireEvent.ended(screen.getByLabelText('Video de HELLO'));
    expect(screen.getByText('2 / 3')).toBeInTheDocument();
    expect(screen.getByLabelText('Video de WATER')).toHaveAttribute('src', '/api/v1/resources/LSM/WATER');
    fireEvent.ended(screen.getByLabelText('Video de WATER'));
    fireEvent.ended(screen.getByLabelText('Video de HELLO'));
    expect(screen.getByText('Secuencia terminada')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Repetir secuencia' }));
    expect(screen.getByText('1 / 3')).toBeInTheDocument();
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('supports pause, resume, previous and next without changing order', async () => {
    const user = userEvent.setup(); render(<SequencePlayer resources={translation.resources} />);
    await user.click(screen.getByRole('button', { name: 'Pausar' }));
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Reproducir' }));
    await user.click(screen.getByRole('button', { name: 'Siguiente' }));
    expect(screen.getByText('2 / 3')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Anterior' }));
    expect(screen.getByText('1 / 3')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Anterior' })).toBeDisabled();
  });
  it('handles blocked autoplay and video decode errors', async () => {
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(new DOMException('Denied', 'NotAllowedError'));
    render(<SequencePlayer resources={translation.resources} />);
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Pulsa Reproducir'));
    fireEvent.error(screen.getByLabelText('Video de HELLO'));
    expect(screen.getByRole('alert')).toHaveTextContent('compatible con tu navegador');
  });
});
