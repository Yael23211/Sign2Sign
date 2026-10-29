import { useEffect } from 'react';

export function isShortcutControl(target: EventTarget | null): boolean {
  return target instanceof Element && !!target.closest(
    'input, textarea, select, button, a, summary, video, audio, [contenteditable]:not([contenteditable="false"]), [role="button"], [role="textbox"], [role="combobox"]',
  );
}

// The handler returns true only when the action is allowed and handled.
export function useShortcuts(handle: (key: string) => boolean) {
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat || event.isComposing || event.ctrlKey || event.altKey || event.metaKey || event.shiftKey
        || isShortcutControl(event.target) || isShortcutControl(document.activeElement)) return;
      if (handle(event.key.toLowerCase())) event.preventDefault();
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, [handle]);
}
