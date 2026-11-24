import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { SequencePlayer } from './SequencePlayer';
import { translation } from '../test/fixtures';
import { UiLanguageContext } from '../i18n/UiLanguage';
import { useShortcuts } from '../hooks/useShortcuts';

describe('ordered sequence playback', () => {
  it('autoplays, advances on ended, stops at end and repeats without HTTP', async () => {
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
    const user = userEvent.setup();
    render(<SequencePlayer resources={translation.resources} />);
    expect(screen.getByText('1 / 3')).toBeInTheDocument();
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
    fireEvent.ended(screen.getByLabelText('Video de HOLA'));
    expect(screen.getByText('2 / 3')).toBeInTheDocument();
    expect(screen.getByLabelText('Video de AGUA')).toHaveAttribute('src', '/api/v1/resources/LSM/WATER');
    fireEvent.ended(screen.getByLabelText('Video de AGUA'));
    fireEvent.ended(screen.getByLabelText('Video de HOLA'));
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
    fireEvent.error(screen.getByLabelText('Video de HOLA'));
    expect(screen.getByRole('alert')).toHaveTextContent('compatible con tu navegador');
  });
});

describe('timeline and focus mode on the existing player', () => {
  const resources = ['FIRST_PERSON_SINGULAR', 'WANT', 'WATER'].map((concept_id, index) => ({
    concept_id, language: 'LSM' as const, url: `/api/v1/resources/LSM/${concept_id}`, position: index + 1, total: 3,
  }));

  it('shows semantic labels and distinguishes actual completed clips from pending ones', () => {
    render(<SequencePlayer resources={resources} />);
    expect(screen.queryByText('FIRST_PERSON_SINGULAR')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'YO, elemento 1 de 3, actual' })).toHaveAttribute('aria-current', 'step');
    expect(screen.getByRole('button', { name: 'QUERER, elemento 2 de 3, pendiente' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'AGUA, elemento 3 de 3, pendiente' })).toBeInTheDocument();
    fireEvent.ended(screen.getByLabelText('Video de YO'));
    expect(screen.getByRole('button', { name: 'YO, elemento 1 de 3, completado' })).toHaveTextContent('✓');
    expect(screen.getByRole('button', { name: 'QUERER, elemento 2 de 3, actual' })).toHaveAttribute('aria-current', 'step');
    expect(screen.getByRole('button', { name: 'AGUA, elemento 3 de 3, pendiente' })).toBeInTheDocument();
  });

  it('selects a timeline clip, pauses, and does not mark skipped clips as played or call HTTP', async () => {
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
    render(<SequencePlayer resources={resources} />);
    await userEvent.click(screen.getByRole('button', { name: 'AGUA, elemento 3 de 3, pendiente' }));
    expect(screen.getByText('3 / 3')).toBeInTheDocument();
    expect(screen.getByLabelText('Video de AGUA')).toHaveAttribute('src', resources[2].url);
    expect(screen.getByRole('button', { name: 'Reproducir' })).toBeInTheDocument();
    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'QUERER, elemento 2 de 3, pendiente' })).toBeInTheDocument();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('opens and closes focus at an intermediate paused clip without recreating video or timeline', async () => {
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
    render(<SequencePlayer resources={resources} />);
    await userEvent.click(screen.getByRole('button', { name: 'QUERER, elemento 2 de 3, pendiente' }));
    const video = screen.getByLabelText('Video de QUERER') as HTMLVideoElement;
    video.currentTime = 1.25;
    const timeline = screen.getByRole('list', { name: 'Secuencia de señas' });
    const plays = vi.mocked(HTMLMediaElement.prototype.play).mock.calls.length;
    await userEvent.click(screen.getByRole('button', { name: 'Modo enfoque' }));
    const dialog = screen.getByRole('dialog', { name: 'Tu mensaje en señas' });
    expect(within(dialog).getByText('2 / 3')).toBeInTheDocument();
    expect(within(dialog).getByLabelText('Video de QUERER')).toBe(video);
    expect(within(dialog).getByRole('list', { name: 'Secuencia de señas' })).toBe(timeline);
    expect(within(dialog).getByRole('button', { name: 'Reproducir' })).toBeInTheDocument();
    expect(video.currentTime).toBe(1.25);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(plays);
    expect(screen.getByRole('button', { name: 'Cerrar modo enfoque' })).toHaveFocus();
    await userEvent.click(screen.getByRole('button', { name: 'Cerrar modo enfoque' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByText('2 / 3')).toBeInTheDocument();
    expect(screen.getByLabelText('Video de QUERER')).toBe(video);
    expect(video.currentTime).toBe(1.25);
    expect(screen.getByRole('button', { name: 'Modo enfoque' })).toHaveFocus();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('keeps automatic advancement and all controls, including local P replay, inside focus mode', async () => {
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher);
    render(<SequencePlayer resources={resources} />);
    await userEvent.click(screen.getByRole('button', { name: 'Modo enfoque' }));
    fireEvent.ended(screen.getByLabelText('Video de YO'));
    expect(screen.getByText('2 / 3')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Pausar' }));
    await userEvent.click(screen.getByRole('button', { name: 'Reproducir' }));
    await userEvent.click(screen.getByRole('button', { name: 'Siguiente' }));
    expect(screen.getByText('3 / 3')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Anterior' }));
    expect(screen.getByText('2 / 3')).toBeInTheDocument();
    fireEvent.ended(screen.getByLabelText('Video de QUERER'));
    fireEvent.ended(screen.getByLabelText('Video de AGUA'));
    expect(screen.getByText('Secuencia terminada')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /actual$/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Repetir secuencia' }));
    expect(screen.getByText('1 / 3')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Siguiente' }));
    (document.activeElement as HTMLElement).blur();
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'p' });
    expect(screen.getByText('1 / 3')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'QUERER, elemento 2 de 3, pendiente' })).toBeInTheDocument();
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('localizes focus UI while the output labels remain tied to the resource language', async () => {
    const { rerender } = render(<UiLanguageContext.Provider value="en"><SequencePlayer resources={resources} /></UiLanguageContext.Provider>);
    expect(screen.getByRole('button', { name: 'YO, item 1 of 3, current' })).toBeInTheDocument();
    expect(screen.queryByText('FIRST_PERSON_SINGULAR')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Focus mode' }));
    const video = screen.getByLabelText('Video of YO');
    rerender(<UiLanguageContext.Provider value="es"><SequencePlayer resources={resources} /></UiLanguageContext.Provider>);
    expect(screen.getByRole('button', { name: 'Cerrar modo enfoque' })).toBeInTheDocument();
    expect(screen.getByLabelText('Video de YO')).toBe(video);
    expect(screen.getByRole('button', { name: 'YO, elemento 1 de 3, actual' })).toBeInTheDocument();
  });

  it('uses English canonical output labels for ASL even in a Spanish interface', () => {
    render(<SequencePlayer resources={[{ ...resources[0], language: 'ASL' }]} />);
    expect(screen.getByRole('button', { name: 'I, elemento 1 de 1, actual' })).toBeInTheDocument();
    expect(screen.getByLabelText('Video de I')).toBeInTheDocument();
  });

  it('traps focus, restores page scroll, and never lets Escape or editing shortcuts reach the background', async () => {
    const backgroundAction = vi.fn();
    function Background() {
      useShortcuts(key => { if (['escape', 'enter', ' ', 'backspace', 't'].includes(key)) { backgroundAction(); return true; } return false; });
      return <button>Background action</button>;
    }
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'auto';
    const { unmount } = render(<><Background /><SequencePlayer resources={resources} /></>);
    const background = screen.getByRole('button', { name: 'Background action' });
    await userEvent.click(screen.getByRole('button', { name: 'Modo enfoque' }));
    expect(background.inert).toBe(true);
    expect(document.body.style.overflow).toBe('hidden');
    const close = screen.getByRole('button', { name: 'Cerrar modo enfoque' });
    fireEvent.keyDown(close, { key: 'Tab', shiftKey: true });
    const repeat = screen.getByRole('button', { name: 'Repetir secuencia' });
    expect(repeat).toHaveFocus();
    fireEvent.keyDown(repeat, { key: 'Tab' }); expect(close).toHaveFocus();
    (document.activeElement as HTMLElement).blur();
    for (const key of ['Escape', 'Enter', ' ', 'Backspace', 't']) fireEvent.keyDown(screen.getByRole('dialog'), { key });
    expect(fireEvent.keyDown(screen.getByRole('dialog'), { key: 't', ctrlKey: true })).toBe(true);
    expect(backgroundAction).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    await userEvent.click(close);
    expect(background.inert).not.toBe(true);
    expect(document.body.style.overflow).toBe('auto');
    expect(screen.getByRole('button', { name: 'Modo enfoque' })).toHaveFocus();
    await userEvent.click(screen.getByRole('button', { name: 'Modo enfoque' }));
    unmount(); expect(document.body.style.overflow).toBe('auto');
    document.body.style.overflow = previous;
  });
});
