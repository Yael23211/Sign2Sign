import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { TranslationResult } from './TranslationResult';
import { translation } from '../test/fixtures';
import { UiLanguageContext } from '../i18n/UiLanguage';

describe('translation presentation and tools', () => {
  it('keeps speech languages independent of interface language and cancels the previous reading', async () => {
    const speak = vi.fn(); const cancel = vi.fn();
    vi.stubGlobal('speechSynthesis', { cancel, getVoices: () => [], speak });
    vi.stubGlobal('SpeechSynthesisUtterance', class { constructor(public text: string) {} lang = ''; });
    const { rerender } = render(<UiLanguageContext.Provider value="en"><TranslationResult result={translation} /></UiLanguageContext.Provider>);
    await userEvent.click(screen.getByRole('button', { name: 'Listen to corrected text' }));
    await userEvent.click(screen.getByRole('button', { name: 'Listen to corrected text' }));
    expect(cancel).toHaveBeenCalledTimes(2);
    expect(speak).toHaveBeenLastCalledWith(expect.objectContaining({ lang: 'en-US', text: 'HELLO' }));
    const previous = speak.mock.calls[0][0];
    previous.onerror({ error: 'canceled' });
    expect(screen.queryByText(/Could not read/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Listen to translation' }));
    expect(speak).toHaveBeenLastCalledWith(expect.objectContaining({ lang: 'es-MX', text: 'HOLA' }));
    rerender(<UiLanguageContext.Provider value="es"><TranslationResult result={translation} /></UiLanguageContext.Provider>);
    await userEvent.click(screen.getByRole('button', { name: 'Escuchar texto corregido' }));
    expect(speak).toHaveBeenLastCalledWith(expect.objectContaining({ lang: 'en-US', text: 'HELLO' }));
    await userEvent.click(screen.getByRole('button', { name: 'Escuchar traducción' }));
    expect(speak).toHaveBeenLastCalledWith(expect.objectContaining({ lang: 'es-MX', text: 'HOLA' }));
  });
  it('degrades clearly when speech synthesis is unavailable', async () => {
    vi.stubGlobal('speechSynthesis', undefined);
    render(<UiLanguageContext.Provider value="en"><TranslationResult result={translation} /></UiLanguageContext.Provider>);
    await userEvent.click(screen.getByRole('button', { name: 'Listen to translation' }));
    expect(screen.getByText('Your browser does not support text-to-speech.')).toBeInTheDocument();
  });
  it('displays unresolved-only as a warning rather than a fatal error', () => {
    render(<TranslationResult result={{ ...translation, concepts: [], resources: [], unresolved: ['ZXQ'], corrected_text: '', translated_text: '' }} />);
    expect(screen.getByText(/No hay recursos visuales disponibles/)).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Video de/)).not.toBeInTheDocument();
  });
  it('keeps valid videos together with unresolved content', () => {
    render(<TranslationResult result={{ ...translation, unresolved: ['XYZ'] }} />);
    expect(screen.getByText(/XYZ/)).toBeInTheDocument();
    expect(screen.getByLabelText('Video de HELLO')).toBeInTheDocument();
  });
  it('keeps textual output when visual preparation fails', () => {
    render(<TranslationResult result={{ ...translation, resources: [], state: 'failed',
      error: { stage: 'SC-06', code: 'visual_preparation_failed', message: 'private' } }} />);
    expect(screen.getByText('HOLA')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('SC-06');
    expect(screen.queryByText('private')).not.toBeInTheDocument();
  });
  it('copies, prints and uses local speech APIs in the correct bridge languages', async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    const speak = vi.fn(); vi.stubGlobal('speechSynthesis', { cancel: vi.fn(), getVoices: () => [], speak });
    vi.stubGlobal('SpeechSynthesisUtterance', class { constructor(public text: string) {} lang = ''; });
    const print = vi.spyOn(window, 'print').mockImplementation(() => {});
    render(<TranslationResult result={translation} />);
    await user.click(screen.getByRole('button', { name: 'Copiar traducción' }));
    expect(writeText).toHaveBeenCalledWith('HOLA');
    await user.click(screen.getByRole('button', { name: 'Escuchar texto corregido' }));
    expect(speak).toHaveBeenLastCalledWith(expect.objectContaining({ lang: 'en-US', text: 'HELLO' }));
    await user.click(screen.getByRole('button', { name: 'Escuchar traducción' }));
    expect(speak).toHaveBeenLastCalledWith(expect.objectContaining({ lang: 'es-MX', text: 'HOLA' }));
    await user.click(screen.getByRole('button', { name: 'Imprimir traducción' }));
    expect(print).toHaveBeenCalledOnce();
  });
});
