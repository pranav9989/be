import React from 'react';
import { useNavigate } from 'react-router-dom';
import ThemeToggle from './ThemeToggle';
import './Header.css';

const Header = ({ user, onLogout, title, showBack = false, onBack }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/login');
  };

  const handleBack = () => {
    if (onBack) {
      onBack();
    } else {
      navigate('/dashboard');
    }
  };

  return (
    <header className="header">
      <div className="header-content">
        <div className="logo" onClick={() => navigate(user ? '/dashboard' : '/login')} style={{ cursor: 'pointer' }}>
          <div className="logo-icon">
            <i className="fas fa-brain"></i>
          </div>
          <div className="logo-text">
            <span className="logo-title">{title || 'InterviewAI'}</span>
          </div>
        </div>

        <div className="header-actions">
          {user && (
            <div className="user-badge">
              <div className="user-avatar">
                {(user.full_name || user.username || 'U')[0].toUpperCase()}
              </div>
              <span className="user-name">{user.full_name || user.username}</span>
            </div>
          )}

          <ThemeToggle />

          {showBack && (
            <button onClick={handleBack} className="header-btn back-btn">
              <i className="fas fa-arrow-left"></i>
              <span>Dashboard</span>
            </button>
          )}

          {user && (
            <button onClick={handleLogout} className="header-btn logout-btn">
              <i className="fas fa-sign-out-alt"></i>
              <span>Logout</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;