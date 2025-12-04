import React from 'react';
import './SessionHistory.css';

const SessionHistory = ({ history = [] }) => {
  if (history.length === 0) {
    return (
      <section className="history-section">
        <h2>Previous Sessions</h2>
        <div className="no-sessions">
          <i className="fas fa-history"></i>
          <p>No previous sessions found.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="history-section">
      <h2>Previous Sessions</h2>
      <div className="history-container">
        {history.map((session, index) => (
          <div key={index} className="history-card">
            <div className="history-header">
              <div>
                <strong>{session.session_type?.toUpperCase() || 'HR'} Interview</strong>
                <div className="history-date">
                  {new Date(session.created_at).toLocaleString()}
                </div>
              </div>
              {session.score && (
                <span className="score-badge">{session.score}%</span>
              )}
            </div>
            <p>{session.questions_count || 0} questions answered</p>
            {session.feedback && (
              <div className="feedback">
                <strong>Feedback:</strong> {session.feedback}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
};

export default SessionHistory;