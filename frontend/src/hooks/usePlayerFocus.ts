import { useEffect, type RefObject } from 'react';
import { isShortcutControl } from './useShortcuts';

export function usePlayerFocus(focused: boolean, container: RefObject<HTMLElement | null>, toggle: RefObject<HTMLButtonElement | null>) {
  useEffect(() => {
    const element = container.current;
    if (!focused || !element) return;
    const previousOverflow = document.body.style.overflow;
    const background: { element: HTMLElement; inert: boolean }[] = [];
    // Keep every ancestor of this same player active, and isolate its siblings.
    let branch: HTMLElement = element;
    while (branch.parentElement && branch !== document.body) {
      for (const sibling of branch.parentElement.children) {
        if (sibling !== branch && sibling instanceof HTMLElement) {
          background.push({ element: sibling, inert: sibling.inert });
          sibling.inert = true;
        }
      }
      branch = branch.parentElement;
    }
    document.body.style.overflow = 'hidden';
    toggle.current?.focus({ preventScroll: true });

    const keepFocus = (event: FocusEvent) => {
      if (event.target instanceof Node && !element.contains(event.target)) toggle.current?.focus({ preventScroll: true });
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.altKey || event.metaKey || event.isComposing) return;
      if (event.key === 'Escape') {
        // Escape already means clear in the main UI. It does not close this view.
        event.preventDefault(); event.stopPropagation(); return;
      }
      if (event.key === 'Tab') {
        const buttons = element.querySelectorAll<HTMLButtonElement>('button:not([disabled])');
        const first = buttons[0]; const last = buttons[buttons.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
        return;
      }
      if (['enter', ' ', 'backspace', 't'].includes(event.key.toLowerCase())
        && !isShortcutControl(event.target) && !isShortcutControl(document.activeElement)) {
        // Do not trigger camera/text-builder shortcuts behind the overlay; P remains local.
        event.preventDefault(); event.stopPropagation();
      }
    };
    window.addEventListener('keydown', onKey, true);
    document.addEventListener('focusin', keepFocus);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      document.removeEventListener('focusin', keepFocus);
      background.forEach(item => { item.element.inert = item.inert; });
      document.body.style.overflow = previousOverflow;
      toggle.current?.focus({ preventScroll: true });
    };
  }, [focused, container, toggle]);
}
