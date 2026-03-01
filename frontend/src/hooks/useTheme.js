import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'interviewai-theme';

/**
 * useTheme — manage light / dark / system theme preference
 * Returns { theme, setTheme, resolvedTheme }
 *   theme         → 'light' | 'dark' | 'system'  (user preference)
 *   resolvedTheme → 'light' | 'dark'              (actual applied)
 */
export function useTheme() {
  const getStoredTheme = () => {
    try {
      return localStorage.getItem(STORAGE_KEY) || 'system';
    } catch {
      return 'system';
    }
  };

  const getSystemTheme = () =>
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';

  const resolve = (pref) => (pref === 'system' ? getSystemTheme() : pref);

  const [theme, setThemeState] = useState(getStoredTheme);
  const [resolvedTheme, setResolvedTheme] = useState(() => resolve(getStoredTheme()));

  const applyTheme = useCallback((pref) => {
    const resolved = resolve(pref);
    document.documentElement.setAttribute('data-theme', resolved);
    setResolvedTheme(resolved);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const setTheme = useCallback(
    (newTheme) => {
      try {
        localStorage.setItem(STORAGE_KEY, newTheme);
      } catch {}
      setThemeState(newTheme);
      applyTheme(newTheme);
    },
    [applyTheme]
  );

  // Apply on mount
  useEffect(() => {
    applyTheme(theme);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Listen for OS preference changes (affects 'system' mode)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      if (getStoredTheme() === 'system') {
        applyTheme('system');
      }
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [applyTheme]);

  return { theme, setTheme, resolvedTheme };
}

export default useTheme;
