import React, { useState, useEffect } from 'react';
import api from '../../services/api'; // using default api to make custom call
import './InterviewHistory.css';

export default function InterviewHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all, technical, mock, agentic, coding
  
  const [selectedSession, setSelectedSession] = useState(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const res = await api.get('/user/history');
      if (res.data.success) {
        setHistory(res.data.history);
      }
    } catch (err) {
      console.error("Failed to load interview history:", err);
    } finally {
      setLoading(false);
    }
  };

  const filteredHistory = history.filter(session => {
    if (filter === 'all') return true;
    if (filter === 'mock' && session.session_type === 'mock') return true;
    if (filter === 'agentic' && session.session_type === 'agentic') return true;
    if (filter === 'debugging' && session.session_type === 'debugging') return true;
    return false;
  });

  const getSessionIcon = (type) => {
    switch (type) {
      case 'hr': return 'fas fa-code'; // Technical Interview
      case 'mock': return 'fas fa-file-invoice';
      case 'agentic': return 'fas fa-brain';
      case 'coding': return 'fas fa-database';
      case 'debugging': return 'fas fa-bug';
      default: return 'fas fa-laptop-code';
    }
  };

  const getSessionName = (type) => {
    switch(type) {
      case 'hr': return 'Technical Chat';
      case 'mock': return 'Mock Interview (Resume)';
      case 'agentic': return 'Voice Agentic Interview';
      case 'coding': return 'Data Science Coding';
      case 'debugging': return 'Code Debugging Practice';
      default: return 'Interview Session';
    }
  };

  const formatDate = (dateString) => {
    let safeDate = dateString;
    if (safeDate && !safeDate.endsWith('Z') && !safeDate.includes('+')) {
      safeDate += 'Z';
    }
    const d = new Date(safeDate);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'short', day: 'numeric', year: 'numeric', 
      hour: 'numeric', minute: '2-digit' 
    }).format(d);
  };

  return (
    <div className="ih-container">
      <div className="ih-header">
        <h2>Interview History & Revision</h2>
        <p>Review your past sessions, AI feedback, and optimal model answers to prepare effectively.</p>
        
        <div className="ih-filters">
          <button className={`ih-filter-btn ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>All Sessions</button>
          <button className={`ih-filter-btn ${filter === 'mock' ? 'active' : ''}`} onClick={() => setFilter('mock')}>
             <i className="fas fa-file-invoice"></i> Mock (Resume)
          </button>
          <button className={`ih-filter-btn ${filter === 'agentic' ? 'active' : ''}`} onClick={() => setFilter('agentic')}>
             <i className="fas fa-brain"></i> Agentic Voice
          </button>
          <button className={`ih-filter-btn ${filter === 'debugging' ? 'active' : ''}`} onClick={() => setFilter('debugging')}>
             <i className="fas fa-bug"></i> Code Debugging
          </button>
        </div>
      </div>

      <div className="ih-content">
        {loading ? (
          <div className="ih-loading"><i className="fas fa-spinner fa-spin"></i> Loading records...</div>
        ) : filteredHistory.length === 0 ? (
          <div className="ih-empty">
             <i className="fas fa-folder-open"></i>
             <p>No interview sessions found for this category.</p>
          </div>
        ) : (
          <div className="ih-grid">
            {filteredHistory.map(session => (
              <div key={session.id} className="ih-card" onClick={() => setSelectedSession(session)}>
                <div className="ih-card-icon">
                  <i className={getSessionIcon(session.session_type)}></i>
                </div>
                <div className="ih-card-body">
                  <h4>{getSessionName(session.session_type)}</h4>
                  <div className="ih-date"><i className="far fa-calendar-alt"></i> {formatDate(session.created_at)}</div>
                  <div className="ih-stats">
                    {session.score !== null && (
                      <span className="ih-score">Score: {Math.round(session.score)}/100</span>
                    )}
                    {session.duration && (
                      <span className="ih-duration"><i className="far fa-clock"></i> {Math.round(session.duration / 60)} min</span>
                    )}
                  </div>
                </div>
                <div className="ih-card-arrow">
                  <i className="fas fa-chevron-right"></i>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedSession && (
        <SessionModal 
          session={selectedSession} 
          onClose={() => setSelectedSession(null)} 
          getSessionName={getSessionName}
          formatDate={formatDate}
        />
      )}
    </div>
  );
}

// ─── Modal to display specific details ──────────────────────────────────────

function SessionModal({ session, onClose, getSessionName, formatDate }) {
  const data = session.data || {};
  const questions = data.questions || [];
  const answers = data.answers || {};

  // For Mock Interviews & Coding, answer format is nested. For old technical chats it's simpler.
  // We'll build a standard data structure to render.

  let renderedQuestions = [];

const renderArrayOfText = (arr) => {
    if (!Array.isArray(arr)) return arr || 'N/A';
    return arr.map(item => {
      if (typeof item === 'string') return item;
      if (typeof item === 'object' && item !== null) {
        return item.missing || item.suggestion || item.keyword || Object.values(item).join(': ');
      }
      return String(item);
    }).join(' • ');
  };

  if (session.session_type === 'mock' || session.session_type === 'agentic') {
     const userAnswers = answers.user_answers || {};
     const evals = answers.evaluations || {};

     renderedQuestions = questions.map((q, i) => {
        const ev = evals[i] || {};
        return (
          <div key={i} className="rev-item">
            <h4 className="rev-q">Q{i+1}: {q.question}</h4>
            <div className="rev-split">
               <div className="rev-user">
                  <h5>Your Answer</h5>
                  <p className="rev-ans">{userAnswers[i] || <em>No answer provided</em>}</p>
                  <div className={`rev-eval grade-${ev.grade}`}>
                     <strong>Grade: {ev.grade} • Score: {ev.score}</strong>
                     <p><strong>Strengths:</strong> {renderArrayOfText(ev.strengths)}</p>
                     <p><strong>Improvements:</strong> {renderArrayOfText(ev.improvements)}</p>
                  </div>
               </div>
               <div className="rev-ai">
                  <h5>Ideal AI Answer</h5>
                  <p className="rev-ideal">{ev.ideal_answer || ev.model_answer || 'N/A'}</p>
               </div>
            </div>
          </div>
        );
     });
  } else if (session.session_type === 'debugging') {
     const challenges = data.challenges || [];
     renderedQuestions = challenges.map((c, i) => (
        <div key={i} className="rev-item debug-rev-item">
           <div className="rev-debug-header">
              <h4 className="rev-q">Challenge {i+1}: {c.topic} ({c.language.toUpperCase()})</h4>
              <span className={`rev-score ${c.ai_score > 0.6 ? 'pass' : 'fail'}`}>
                 Score: {Math.round(c.ai_score * 100)}%
              </span>
           </div>
           
           <div className="rev-debug-content">
              <div className="code-block-rev">
                 <h5>Buggy Code</h5>
                 <pre><code>{c.buggy_code}</code></pre>
              </div>
              
              <div className="explanation-rev">
                 <h5>Your Explanation</h5>
                 <p>"{c.user_explanation || 'No explanation provided'}"</p>
              </div>

              <div className="feedback-rev">
                 <h5>AI Analysis</h5>
                 <p>{c.ai_feedback}</p>
              </div>

              <div className="code-block-rev fixed">
                 <h5>Corrected Code</h5>
                 <pre><code>{c.correct_code}</code></pre>
              </div>
           </div>
        </div>
     ));
  } else {
     // Legacy Technical/HR fallback
     const qList = Array.isArray(questions) ? questions : [];
     renderedQuestions = qList.map((q, i) => (
       <div key={i} className="rev-item">
         <h4 className="rev-q">Q{i+1}: {q.question || q}</h4>
         <div className="rev-split">
            <div className="rev-user">
               <h5>Your Answer</h5>
               <p className="rev-ans">{answers[i] || <em>N/A</em>}</p>
            </div>
         </div>
       </div>
     ));
  }

  return (
    <div className="ih-modal-backdrop" onClick={onClose}>
      <div className="ih-modal" onClick={e => e.stopPropagation()}>
        <div className="ih-modal-header">
           <div className="modal-title-area">
             <h3>{getSessionName(session.session_type)}</h3>
             <span className="ih-modal-date">{formatDate(session.created_at)}</span>
           </div>
           <button className="ih-close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="ih-modal-body">
           <div className="rev-summary">
              {session.score !== null && <span>Overall Score: <strong>{Math.round(session.score)}/100</strong></span>}
           </div>
           <div className="rev-questions">
              {renderedQuestions.length > 0 ? renderedQuestions : <p>No detailed data saved for this session.</p>}
           </div>
        </div>
      </div>
    </div>
  );
}
