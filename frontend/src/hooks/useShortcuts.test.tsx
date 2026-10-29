import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useShortcuts } from './useShortcuts';

function Harness({ handle }: { handle: (key: string) => boolean }) {
  useShortcuts(handle);
  return <><input aria-label="input" /><textarea aria-label="textarea" /><select aria-label="select"><option>A</option></select>
    <button><span>button</span></button><div contentEditable suppressContentEditableWarning data-testid="editable">Editable</div>
    <a href="#link">link</a><details><summary>summary</summary></details></>;
}

it.each(['input', 'textarea', 'select', 'button', 'editable', 'link', 'summary'])(
  'does not intercept shortcuts in %s or its children', target => {
    const handle = vi.fn(() => true); render(<Harness handle={handle} />);
    const element = target === 'editable' ? screen.getByTestId(target) : target === 'button'
      ? screen.getByText('button') : target === 'link' || target === 'summary'
        ? screen.getByText(target) : screen.getByLabelText(target);
    for (const key of ['Enter', ' ', 'Backspace', 'Escape', 't', 'p']) {
      expect(fireEvent.keyDown(element, { key })).toBe(true);
    }
    expect(handle).not.toHaveBeenCalled();
  });

it('ignores a focused control even for window events and removes the listener on unmount', () => {
  const handle = vi.fn(() => true); const { unmount } = render(<Harness handle={handle} />);
  screen.getByLabelText('input').focus(); fireEvent.keyDown(window, { key: 't' });
  expect(handle).not.toHaveBeenCalled();
  (document.activeElement as HTMLElement).blur();
  expect(fireEvent.keyDown(document.body, { key: 't' })).toBe(false);
  expect(handle).toHaveBeenCalledOnce();
  unmount(); fireEvent.keyDown(window, { key: 'p' }); expect(handle).toHaveBeenCalledOnce();
});

it.each([{ repeat: true }, { isComposing: true }, { ctrlKey: true }, { altKey: true }, { metaKey: true }, { shiftKey: true }])(
  'ignores key repeats, composition and modifiers: %j', options => {
    const handle = vi.fn(() => true); render(<Harness handle={handle} />);
    fireEvent.keyDown(document.body, { key: 't', ...options }); expect(handle).not.toHaveBeenCalled();
  });

it('keeps native behavior when an action is unavailable', () => {
  render(<Harness handle={() => false} />);
  expect(fireEvent.keyDown(document.body, { key: ' ' })).toBe(true);
});
