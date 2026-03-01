import React from 'react';
import './WelcomeMessage.css';

const WelcomeMessage = () => {
  return (
    <div className="welcome-message">
      <div className="welcome-icon">
        <i className="fas fa-brain"></i>
      </div>
      <div>
        <h3>Welcome to Technical Interview</h3>
        <p>Ask me anything about Database Management Systems (DBMS), Object-Oriented Programming (OOPs), or Operating Systems (OS).</p>
      </div>
      <div className="welcome-topics">
        <span className="welcome-topic-tag">📊 DBMS</span>
        <span className="welcome-topic-tag">🧬 OOPs</span>
        <span className="welcome-topic-tag">⚙️ Operating Systems</span>
        <span className="welcome-topic-tag">✅ AI-Powered</span>
        <span className="welcome-topic-tag">📚 Source References</span>
      </div>
    </div>
  );
};

export default WelcomeMessage;