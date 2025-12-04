import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Header.css';

const Header = ({ user, onLogout, title, showBack = false }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/login');
  };

  const handleBack = () => {
    navigate('/dashboard');
  };

  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <i className="fas fa-brain"></i>
          <h1>{title}</h1>
        </div>
        
        <div className="header-actions">
          {user && (
            <div className="user-info">
              <span>Welcome, {user.full_name || user.username}</span>
            </div>
          )}
          
          {showBack && (
            <button onClick={handleBack} className="back-btn">
              <i className="fas fa-arrow-left"></i> Back to Dashboard
            </button>
          )}
          
          {user && (
            <button onClick={handleLogout} className="logout-btn">
              <i className="fas fa-sign-out-alt"></i> Logout
            </button>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;