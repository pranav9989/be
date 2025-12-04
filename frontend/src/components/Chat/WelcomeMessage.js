import React from 'react';
import './WelcomeMessage.css';

const WelcomeMessage = () => {
  return (
    <div className="welcome-message">
      <div className="welcome-content">
        <i className="fas fa-robot"></i>
        <h2>Welcome to CS Interview Assistant!</h2>
        <p>Ask me anything about Database Management Systems (DBMS), Object-Oriented Programming (OOPs), or Operating Systems (OS).</p>
        <div className="features">
          <span className="feature"><i className="fas fa-check"></i> AI-Powered Responses</span>
          <span className="feature"><i className="fas fa-check"></i> Topic Detection</span>
          <span className="feature"><i className="fas fa-check"></i> Source References</span>
        </div>
      </div>
    </div>
  );
};

export default WelcomeMessage;