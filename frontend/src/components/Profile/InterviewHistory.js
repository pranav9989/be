import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import './InterviewHistory.css';

export default function InterviewHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  const [selectedSession, setSelectedSession] = useState(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const res = await api.get('/user/history');
      if (res.data.success) {
        console.log('📊 Raw history data:', res.data.history);
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
      case 'hr': return 'fas fa-code';
      case 'mock': return 'fas fa-file-invoice';
      case 'agentic': return 'fas fa-brain';
      case 'coding': return 'fas fa-database';
      case 'debugging': return 'fas fa-bug';
      default: return 'fas fa-laptop-code';
    }
  };

  const getSessionName = (type) => {
    switch (type) {
      case 'hr': return 'Technical Chat';
      case 'mock': return 'Mock Interview (Resume)';
      case 'agentic': return 'Voice Agentic Interview';
      case 'coding': return 'Data Science Coding';
      case 'debugging': return 'Code Debugging Practice';
      default: return 'Interview Session';
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
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

  // 🔥 Helper to get session metrics from speech_metrics JSON
  const getSessionMetrics = (session) => {
    let metrics = null;

    if (session.metrics) {
      metrics = session.metrics;
    }
    else if (session.data?.metrics) {
      metrics = session.data.metrics;
    }
    else if (session.speech_metrics) {
      try {
        metrics = typeof session.speech_metrics === 'string'
          ? JSON.parse(session.speech_metrics)
          : session.speech_metrics;
      } catch (e) {
        console.log('Error parsing speech_metrics:', e);
      }
    }

    return metrics;
  };

  // 🔥 Format time nicely
  const formatDuration = (seconds) => {
    if (!seconds) return null;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
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
            {filteredHistory.map(session => {
              const metrics = getSessionMetrics(session);

              return (
                <div key={session.id} className="ih-card" onClick={() => setSelectedSession(session)}>
                  <div className="ih-card-icon">
                    <i className={getSessionIcon(session.session_type)}></i>
                  </div>
                  <div className="ih-card-body">
                    <h4>{getSessionName(session.session_type)}</h4>
                    <div className="ih-date"><i className="far fa-calendar-alt"></i> {formatDate(session.created_at)}</div>

                    {/* 🔥 SESSION SCORE */}
                    {session.score ? (
                      <div className="ih-stats">
                        {session.duration && (
                          <span className="ih-duration"><i className="far fa-clock"></i> {Math.round(session.duration / 60)} min</span>
                        )}
                      </div>
                    ) : metrics?.avg_semantic_similarity ? (
                      <div className="ih-stats">
                      </div>
                    ) : null}

                    {/* 🔥 SESSION METRICS PREVIEW */}
                    {metrics && (
                      <div className="session-metrics-preview" style={{
                        marginTop: '0.75rem',
                        padding: '0.5rem',
                        background: 'rgba(0,0,0,0.2)',
                        borderRadius: '6px',
                        fontSize: '0.75rem',
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '0.25rem 0.5rem'
                      }}>
                        {metrics.wpm > 0 && (
                          <div><span style={{ opacity: 0.7 }}>⚡ WPM:</span> <strong>{Math.round(metrics.wpm)}</strong></div>
                        )}
                        {metrics.speaking_time > 0 && (
                          <div><span style={{ opacity: 0.7 }}>🎤 Speaking:</span> <strong>{formatDuration(metrics.speaking_time)}</strong></div>
                        )}
                        {metrics.silence_time > 0 && (
                          <div><span style={{ opacity: 0.7 }}>🤔 Thinking:</span> <strong>{formatDuration(metrics.silence_time)}</strong></div>
                        )}
                        {metrics.speaking_ratio > 0 && (
                          <div><span style={{ opacity: 0.7 }}>📊 Speak Ratio:</span> <strong>{Math.round(metrics.speaking_ratio * 100)}%</strong></div>
                        )}
                        {metrics.avg_response_latency > 0 && (
                          <div><span style={{ opacity: 0.7 }}>⏱️ Latency:</span> <strong>{metrics.avg_response_latency.toFixed(2)}s</strong></div>
                        )}
                        {metrics.articulation_rate > 0 && (
                          <div><span style={{ opacity: 0.7 }}>📝 Words/s:</span> <strong>{metrics.articulation_rate.toFixed(2)}</strong></div>
                        )}
                        {metrics.avg_semantic_similarity > 0 && (
                          <div><span style={{ opacity: 0.7 }}>📚 Semantic:</span> <strong>{Math.round(metrics.avg_semantic_similarity * 100)}%</strong></div>
                        )}
                        {metrics.avg_keyword_coverage > 0 && (
                          <div><span style={{ opacity: 0.7 }}>🔑 Keyword:</span> <strong>{Math.round(metrics.avg_keyword_coverage * 100)}%</strong></div>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="ih-card-arrow">
                    <i className="fas fa-chevron-right"></i>
                  </div>
                </div>
              );
            })}
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

// ─── Modal to display ALL session metrics with COACHING FEEDBACK ────────────

function SessionModal({ session, onClose, getSessionName, formatDate }) {
  const data = session.data || {};
  const questions = data.questions || [];
  const answers = data.answers || {};

  // 🔥 CRITICAL FIX: Get feedback from multiple possible locations
  const coachingFeedback = session.feedback || data.feedback || null;

  // 🔥 Debug log to verify feedback exists
  React.useEffect(() => {
    if (coachingFeedback) {
      console.log('✅ Feedback loaded for session:', session.id);
      console.log('📝 Feedback preview:', coachingFeedback.substring(0, 100));
    } else {
      console.log('⚠️ No feedback found for session:', session.id);
    }
  }, [coachingFeedback, session.id]);

  // 🔥 Function to render coaching feedback (improved)
  const renderCoachingFeedback = (feedbackText) => {
    if (!feedbackText || feedbackText === 'null' || feedbackText === 'undefined') {
      return (
        <div className="coaching-feedback-modal" style={{
          background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
          padding: '1.5rem',
          borderRadius: '16px',
          marginBottom: '2rem',
          border: '1px solid rgba(212, 168, 83, 0.3)',
          textAlign: 'center'
        }}>
          <h4 style={{ margin: '0 0 1rem 0', color: '#a78bfa' }}>
            <i className="fas fa-comment-dots"></i> Coaching Feedback
          </h4>
          <p style={{ color: '#888' }}>No feedback available for this session yet.</p>
          <p style={{ color: '#666', fontSize: '0.8rem', marginTop: '0.5rem' }}>
            Feedback is generated automatically after each session.
          </p>
        </div>
      );
    }

    // Parse sections with improved regex
    const sections = [];

    // Try to parse markdown sections
    const sectionRegex = /##\s*([^\n]+)\n([\s\S]*?)(?=##|$)/g;
    let match;

    while ((match = sectionRegex.exec(feedbackText)) !== null) {
      const title = match[1].trim();
      const content = match[2].trim();
      if (title && content) {
        sections.push({ title, content });
      }
    }

    // If no sections found, try alternative parsing
    if (sections.length === 0) {
      // Split by lines and look for headers
      const lines = feedbackText.split('\n');
      let currentSection = null;
      let currentContent = [];

      for (const line of lines) {
        if (line.startsWith('##')) {
          if (currentSection && currentContent.length) {
            sections.push({ title: currentSection, content: currentContent.join('\n') });
          }
          currentSection = line.replace(/^##\s*/, '').trim();
          currentContent = [];
        } else if (currentSection && line.trim()) {
          currentContent.push(line);
        }
      }

      if (currentSection && currentContent.length) {
        sections.push({ title: currentSection, content: currentContent.join('\n') });
      }
    }

    // If still no sections, treat entire text as one section
    if (sections.length === 0 && feedbackText.trim()) {
      sections.push({ title: 'Coaching Insights', content: feedbackText });
    }

    const getIconForTitle = (title) => {
      const t = title.toLowerCase();
      if (t.includes('vocal') || t.includes('voice') || t.includes('delivery')) return '🗣️';
      if (t.includes('technical') || t.includes('content')) return '📚';
      if (t.includes('response') || t.includes('flow')) return '⏱️';
      if (t.includes('practice') || t.includes('exercise')) return '🎯';
      if (t.includes('weak') || t.includes('missing') || t.includes('gap')) return '⚠️';
      if (t.includes('summary') || t.includes('overview')) return '📊';
      return '📌';
    };

    const getColorForTitle = (title) => {
      const t = title.toLowerCase();
      if (t.includes('vocal') || t.includes('voice')) return '#f472b6';
      if (t.includes('technical') || t.includes('content')) return '#a78bfa';
      if (t.includes('response') || t.includes('flow')) return '#6ee7b7';
      if (t.includes('practice') || t.includes('exercise')) return '#f59e0b';
      if (t.includes('weak') || t.includes('missing')) return '#ef4444';
      return '#10b981';
    };

    return (
      <div className="coaching-feedback-modal" style={{
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        padding: '1.5rem',
        borderRadius: '16px',
        marginBottom: '2rem',
        border: '1px solid rgba(212, 168, 83, 0.3)'
      }}>
        <h4 style={{
          margin: '0 0 1rem 0',
          color: '#a78bfa',
          fontSize: '1.1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <i className="fas fa-comment-dots"></i>
          AI-Powered Coaching Feedback
        </h4>

        {sections.map((section, idx) => {
          const icon = getIconForTitle(section.title);
          const borderColor = getColorForTitle(section.title);
          const lines = section.content.split('\n');

          return (
            <div key={idx} style={{
              marginBottom: '1.25rem',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '12px',
              padding: '1rem',
              border: `1px solid ${borderColor}40`
            }}>
              <h5 style={{
                margin: '0 0 0.75rem 0',
                color: borderColor,
                fontSize: '0.9rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                borderBottom: `1px solid ${borderColor}40`,
                paddingBottom: '0.5rem'
              }}>
                <span style={{ fontSize: '1.1rem' }}>{icon}</span>
                {section.title}
              </h5>

              {lines.map((line, lineIdx) => {
                if (!line.trim()) return null;

                // Bold text handling
                const processedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

                // Bullet points
                if (line.trim().startsWith('-') || line.trim().startsWith('*') || line.trim().startsWith('✓') || line.trim().startsWith('•')) {
                  const bulletText = line.replace(/^[-*✓•]\s*/, '').trim();
                  return (
                    <div key={lineIdx} style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.5rem',
                      marginBottom: '0.5rem',
                      padding: '0.25rem 0'
                    }}>
                      <span style={{ color: '#10b981', fontSize: '0.8rem', marginTop: '2px' }}>✓</span>
                      <span style={{ color: '#ccc', fontSize: '0.85rem', lineHeight: '1.5', flex: 1 }}
                        dangerouslySetInnerHTML={{ __html: bulletText }} />
                    </div>
                  );
                }

                // Numbered items
                if (/^\d+\./.test(line.trim())) {
                  const numMatch = line.match(/^(\d+)\.\s*(.*)/);
                  if (numMatch) {
                    return (
                      <div key={lineIdx} style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '0.5rem',
                        marginBottom: '0.5rem',
                        padding: '0.25rem 0'
                      }}>
                        <span style={{ color: '#f59e0b', fontSize: '0.8rem', fontWeight: 'bold', minWidth: '24px' }}>{numMatch[1]}</span>
                        <span style={{ color: '#ccc', fontSize: '0.85rem', lineHeight: '1.5', flex: 1 }}
                          dangerouslySetInnerHTML={{ __html: numMatch[2] }} />
                      </div>
                    );
                  }
                }

                // Regular text (descriptions)
                if (line.trim()) {
                  return (
                    <p key={lineIdx} style={{
                      color: '#aaa',
                      fontSize: '0.85rem',
                      lineHeight: '1.6',
                      margin: '0 0 0.5rem 0',
                      paddingLeft: '0.5rem',
                      borderLeft: `2px solid ${borderColor}`
                    }}
                      dangerouslySetInnerHTML={{ __html: processedLine }} />
                  );
                }

                return null;
              })}
            </div>
          );
        })}
      </div>
    );
  };

  // 🔥 Get session metrics
  const getSessionMetrics = (session) => {
    let metrics = null;

    if (session.metrics) {
      metrics = session.metrics;
    } else if (session.data?.metrics) {
      metrics = session.data.metrics;
    } else if (session.speech_metrics) {
      try {
        metrics = typeof session.speech_metrics === 'string'
          ? JSON.parse(session.speech_metrics)
          : session.speech_metrics;
      } catch (e) {
        console.log('Error parsing speech_metrics:', e);
      }
    }

    return metrics;
  };

  const metrics = getSessionMetrics(session);

  // 🔥 Format duration
  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  let renderedQuestions = [];

  if (session.session_type === 'mock' || session.session_type === 'agentic') {
    const userAnswers = answers.user_answers || {};
    const evals = answers.evaluations || {};

    renderedQuestions = questions.map((q, i) => {
      const ev = evals[i] || {};
      return (
        <div key={i} className="rev-item">
          <h4 className="rev-q">Q{i + 1}: {q.question}</h4>
          <div className="rev-split">
            <div className="rev-user">
              <h5>Your Answer</h5>
              <p className="rev-ans">{userAnswers[i] || <em>No answer provided</em>}</p>
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
          <h4 className="rev-q">Challenge {i + 1}: {c.topic} ({c.language.toUpperCase()})</h4>
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
    const qList = Array.isArray(questions) ? questions : [];
    renderedQuestions = qList.map((q, i) => (
      <div key={i} className="rev-item">
        <h4 className="rev-q">Q{i + 1}: {q.question || q}</h4>
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

          {/* 🔥 DISPLAY COACHING FEEDBACK - MOVED TO TOP FOR VISIBILITY */}
          {renderCoachingFeedback(coachingFeedback)}

          {/* 🔥 SESSION METRICS SUMMARY */}
          {metrics && (
            <div className="session-metrics-summary" style={{
              background: 'linear-gradient(135deg, #1a1a2e, #16213e)',
              padding: '1.5rem',
              borderRadius: '12px',
              marginBottom: '2rem',
              color: 'white'
            }}>
              <h4 style={{ margin: '0 0 1rem 0', color: '#a78bfa' }}>📊 Session Performance Metrics</h4>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                {/* Row 1 */}
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Session Duration</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{formatDuration(metrics.session_duration)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Speaking Time</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#6ee7b7' }}>{formatDuration(metrics.speaking_time)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Words Per Minute</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{Math.round(metrics.wpm)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Questions</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{metrics.questions_answered || 0}</div>
                </div>

                {/* Row 2 */}
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Speaking Ratio</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{Math.round(metrics.speaking_ratio * 100)}%</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Response Latency</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{metrics.avg_response_latency?.toFixed(2)}s</div>
                </div>
              </div>
            </div>
          )}

          <div className="rev-questions">
            <h4 style={{ marginBottom: '1rem' }}>📝 Question & Answer Review</h4>
            {renderedQuestions.length > 0 ? renderedQuestions : <p>No detailed data saved for this session.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}