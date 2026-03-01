import React, { useState, useRef, useEffect } from 'react';
import { Sun, Moon, Monitor } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import './ThemeToggle.css';

const THEMES = [
  { key: 'light',  Icon: Sun,     label: 'Light' },
  { key: 'dark',   Icon: Moon,    label: 'Dark'  },
  { key: 'system', Icon: Monitor, label: 'System' },
];

const ThemeToggle = () => {
  const { theme, setTheme } = useTheme();
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef(null);

  const current = THEMES.find((t) => t.key === theme) || THEMES[1];
  const { Icon } = current;

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setShowMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div className="theme-toggle-wrapper" ref={menuRef}>
      <button
        className="theme-toggle-btn"
        onClick={() => setShowMenu((prev) => !prev)}
        aria-label={`Current theme: ${current.label}. Click to switch.`}
        title={`Theme: ${current.label}`}
      >
        <span className="theme-icon-wrap">
          <Icon size={17} strokeWidth={2} />
        </span>
        <span className="theme-label">{current.label}</span>
      </button>

      {showMenu && (
        <div className="theme-menu" role="menu">
          {THEMES.map(({ key, Icon: ItemIcon, label }) => (
            <button
              key={key}
              className={`theme-option${theme === key ? ' active' : ''}`}
              onClick={() => { setTheme(key); setShowMenu(false); }}
              role="menuitem"
            >
              <ItemIcon size={15} strokeWidth={2} />
              <span>{label}</span>
              {theme === key && <span className="theme-check">✓</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default ThemeToggle;
