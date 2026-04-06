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

  // 🔥 Get coaching feedback
  const coachingFeedback = session.feedback || data.feedback || null;

  // 🔥 NEW: Fetch detailed question history from backend
  const [detailedQuestions, setDetailedQuestions] = useState([]);
  const [loadingDetails, setLoadingDetails] = useState(false);

  useEffect(() => {
    // Fetch detailed question history for agentic sessions
    if (session.session_type === 'agentic' && session.id) {
      fetchDetailedHistory();
    }
  }, [session.id, session.session_type]);

  const fetchDetailedHistory = async () => {
    setLoadingDetails(true);
    try {
      // Try to get from session data first
      if (session.detailed_questions) {
        setDetailedQuestions(session.detailed_questions);
      } else {
        // Fetch from API endpoint (you may need to create this)
        const res = await api.get(`/user/session/${session.id}/questions`);
        if (res.data.success) {
          setDetailedQuestions(res.data.questions);
        }
      }
    } catch (err) {
      console.error('Failed to fetch detailed questions:', err);
    } finally {
      setLoadingDetails(false);
    }
  };

  React.useEffect(() => {
    if (coachingFeedback) {
      console.log('✅ Feedback loaded for session:', session.id);
    }
  }, [coachingFeedback, session.id]);

  // Helper to get icon for difficulty
  const getDifficultyIcon = (difficulty) => {
    switch (difficulty?.toLowerCase()) {
      case 'easy': return '🟢';
      case 'medium': return '🟡';
      case 'hard': return '🔴';
      default: return '⚪';
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty?.toLowerCase()) {
      case 'easy': return '#4ade80';
      case 'medium': return '#fbbf24';
      case 'hard': return '#ef4444';
      default: return '#6b7280';
    }
  };

  // Render coaching feedback
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
        </div>
      );
    }

    const sections = [];
    const sectionRegex = /##\s*([^\n]+)\n([\s\S]*?)(?=##|$)/g;
    let match;

    while ((match = sectionRegex.exec(feedbackText)) !== null) {
      const title = match[1].trim();
      const content = match[2].trim();
      if (title && content) {
        sections.push({ title, content });
      }
    }

    if (sections.length === 0 && feedbackText.trim()) {
      sections.push({ title: 'Coaching Insights', content: feedbackText });
    }

    const getIconForTitle = (title) => {
      const t = title.toLowerCase();
      if (t.includes('vocal') || t.includes('voice') || t.includes('delivery')) return '🗣️';
      if (t.includes('technical') || t.includes('content')) return '📚';
      if (t.includes('response') || t.includes('flow')) return '⏱️';
      if (t.includes('practice') || t.includes('exercise')) return '🎯';
      return '📌';
    };

    const getColorForTitle = (title) => {
      const t = title.toLowerCase();
      if (t.includes('vocal') || t.includes('voice')) return '#f472b6';
      if (t.includes('technical') || t.includes('content')) return '#a78bfa';
      if (t.includes('response') || t.includes('flow')) return '#6ee7b7';
      if (t.includes('practice') || t.includes('exercise')) return '#f59e0b';
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
                const processedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

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

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  // 🔥 RENDER QUESTIONS WITH FULL DETAILS (Topic, Subtopic, Difficulty, Missing Concepts)
  const renderDetailedQuestions = () => {
    if (loadingDetails) {
      return <div style={{ textAlign: 'center', padding: '2rem' }}>Loading detailed questions...</div>;
    }

    // Use detailedQuestions if available, otherwise fall back to session data
    const questionsToRender = detailedQuestions.length > 0 ? detailedQuestions : [];

    if (questionsToRender.length === 0) {
      // Fallback: Show basic questions from session data
      const qList = Array.isArray(questions) ? questions : [];
      const userAnswers = answers.user_answers || {};
      const evals = answers.evaluations || {};

      return qList.map((q, i) => {
        const ev = evals[i] || {};
        return (
          <div key={i} className="rev-item" style={{
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '12px',
            padding: '1.5rem',
            marginBottom: '1rem',
            borderLeft: '3px solid #a78bfa'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <h4 className="rev-q" style={{ margin: 0, color: '#e2e8f0' }}>Q{i + 1}: {q.question}</h4>
            </div>

            <div className="rev-split" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div className="rev-user">
                <h5 style={{ color: '#a78bfa', marginBottom: '0.5rem' }}>Your Answer</h5>
                <p className="rev-ans" style={{ color: '#ccc', lineHeight: '1.6' }}>{userAnswers[i] || <em>No answer provided</em>}</p>
              </div>
              <div className="rev-ai">
                <h5 style={{ color: '#a78bfa', marginBottom: '0.5rem' }}>Ideal AI Answer</h5>
                <p className="rev-ideal" style={{ color: '#ccc', lineHeight: '1.6' }}>{ev.ideal_answer || ev.model_answer || 'N/A'}</p>
              </div>
            </div>
          </div>
        );
      });
    }

    // Render detailed questions with topic, subtopic, difficulty, and missing concepts
    return questionsToRender.map((q, i) => (
      <div key={i} className="rev-item" style={{
        background: 'rgba(255,255,255,0.05)',
        borderRadius: '12px',
        padding: '1.5rem',
        marginBottom: '1rem',
        borderLeft: `3px solid ${getDifficultyColor(q.difficulty)}`
      }}>
        {/* Header with metadata */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem',
          flexWrap: 'wrap',
          gap: '0.5rem',
          paddingBottom: '0.75rem',
          borderBottom: '1px solid rgba(255,255,255,0.1)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span style={{
              background: 'rgba(167, 139, 250, 0.2)',
              padding: '0.25rem 0.75rem',
              borderRadius: '20px',
              fontSize: '0.75rem',
              color: '#a78bfa'
            }}>
              <i className="fas fa-folder"></i> {q.topic || 'Unknown Topic'}
            </span>
            {q.subtopic && (
              <span style={{
                background: 'rgba(110, 231, 183, 0.2)',
                padding: '0.25rem 0.75rem',
                borderRadius: '20px',
                fontSize: '0.75rem',
                color: '#6ee7b7'
              }}>
                <i className="fas fa-tag"></i> {q.subtopic}
              </span>
            )}
            <span style={{
              background: `rgba(${q.difficulty === 'easy' ? '74, 222, 128' : q.difficulty === 'medium' ? '251, 191, 36' : '239, 68, 68'}, 0.2)`,
              padding: '0.25rem 0.75rem',
              borderRadius: '20px',
              fontSize: '0.75rem',
              color: getDifficultyColor(q.difficulty)
            }}>
              {getDifficultyIcon(q.difficulty)} {q.difficulty?.toUpperCase() || 'MEDIUM'}
            </span>
          </div>
          <span style={{
            fontSize: '0.7rem',
            color: '#888'
          }}>
            <i className="far fa-clock"></i> {q.response_time ? `${q.response_time.toFixed(1)}s` : 'N/A'}
          </span>
        </div>

        {/* Question */}
        <h4 className="rev-q" style={{ margin: '0 0 1rem 0', color: '#e2e8f0', fontSize: '1rem' }}>
          {i + 1}. {q.question}
        </h4>

        {/* Scores */}
        {(q.semantic_score > 0 || q.keyword_score > 0) && (
          <div style={{
            display: 'flex',
            gap: '1rem',
            marginBottom: '1rem',
            padding: '0.5rem',
            background: 'rgba(0,0,0,0.3)',
            borderRadius: '8px'
          }}>
            {q.semantic_score > 0 && (
              <span style={{ fontSize: '0.75rem', color: '#a78bfa' }}>
                <i className="fas fa-chart-line"></i> Semantic: {Math.round(q.semantic_score * 100)}%
              </span>
            )}
            {q.keyword_score > 0 && (
              <span style={{ fontSize: '0.75rem', color: '#6ee7b7' }}>
                <i className="fas fa-key"></i> Keyword: {Math.round(q.keyword_score * 100)}%
              </span>
            )}
          </div>
        )}

        {/* Answer Section */}
        <div className="rev-split" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div className="rev-user">
            <h5 style={{ color: '#a78bfa', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <i className="fas fa-user"></i> Your Answer
            </h5>
            <p className="rev-ans" style={{ color: '#ccc', lineHeight: '1.6', fontSize: '0.85rem' }}>
              {q.answer || q.user_answer || <em>No answer provided</em>}
            </p>
          </div>
          <div className="rev-ai">
            <h5 style={{ color: '#6ee7b7', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <i className="fas fa-robot"></i> Ideal AI Answer
            </h5>
            <p className="rev-ideal" style={{ color: '#ccc', lineHeight: '1.6', fontSize: '0.85rem' }}>
              {q.expected_answer || 'N/A'}
            </p>
          </div>
        </div>

        {/* 🔥 MISSING CONCEPTS SECTION */}
        {q.missing_concepts && q.missing_concepts.length > 0 && (
          <div style={{
            marginTop: '1rem',
            padding: '0.75rem',
            background: 'rgba(239, 68, 68, 0.1)',
            borderRadius: '8px',
            borderLeft: '3px solid #ef4444'
          }}>
            <h5 style={{ color: '#ef4444', marginBottom: '0.5rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <i className="fas fa-exclamation-triangle"></i> Missing Concepts
            </h5>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {q.missing_concepts.map((concept, idx) => (
                <span key={idx} style={{
                  background: 'rgba(239, 68, 68, 0.2)',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '16px',
                  fontSize: '0.7rem',
                  color: '#fca5a5'
                }}>
                  {concept}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* 🔥 SAMPLED CONCEPTS (What was asked) */}
        {q.sampled_concepts && q.sampled_concepts.length > 0 && (
          <div style={{
            marginTop: '0.75rem',
            padding: '0.5rem 0.75rem',
            background: 'rgba(167, 139, 250, 0.1)',
            borderRadius: '8px'
          }}>
            <h5 style={{ color: '#a78bfa', marginBottom: '0.5rem', fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <i className="fas fa-microchip"></i> Concepts Tested
            </h5>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {q.sampled_concepts.map((concept, idx) => (
                <span key={idx} style={{
                  background: 'rgba(167, 139, 250, 0.2)',
                  padding: '0.2rem 0.6rem',
                  borderRadius: '16px',
                  fontSize: '0.65rem',
                  color: '#c4b5fd'
                }}>
                  {concept}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    ));
  };

  return (
    <div className="ih-modal-backdrop" onClick={onClose}>
      <div className="ih-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: '900px', width: '90%' }}>
        <div className="ih-modal-header">
          <div className="modal-title-area">
            <h3>{getSessionName(session.session_type)}</h3>
            <span className="ih-modal-date">{formatDate(session.created_at)}</span>
          </div>
          <button className="ih-close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="ih-modal-body" style={{ maxHeight: '80vh', overflowY: 'auto' }}>

          {/* Coaching Feedback */}
          {renderCoachingFeedback(coachingFeedback)}

          {/* Session Metrics */}
          {metrics && (
            <div className="session-metrics-summary" style={{
              background: 'linear-gradient(135deg, #1a1a2e, #16213e)',
              padding: '1rem',
              borderRadius: '12px',
              marginBottom: '1.5rem'
            }}>
              <h4 style={{ margin: '0 0 0.75rem 0', color: '#a78bfa', fontSize: '0.9rem' }}>📊 Session Metrics</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '0.75rem' }}>
                <div><div style={{ fontSize: '0.65rem', opacity: 0.7 }}>Duration</div><div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{formatDuration(metrics.session_duration)}</div></div>
                <div><div style={{ fontSize: '0.65rem', opacity: 0.7 }}>Speaking</div><div style={{ fontSize: '0.9rem', fontWeight: 'bold', color: '#6ee7b7' }}>{formatDuration(metrics.speaking_time)}</div></div>
                <div><div style={{ fontSize: '0.65rem', opacity: 0.7 }}>WPM</div><div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{Math.round(metrics.wpm)}</div></div>
                <div><div style={{ fontSize: '0.65rem', opacity: 0.7 }}>Questions</div><div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{metrics.questions_answered || 0}</div></div>
                <div><div style={{ fontSize: '0.65rem', opacity: 0.7 }}>Speaking Ratio</div><div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{Math.round(metrics.speaking_ratio * 100)}%</div></div>
                <div><div style={{ fontSize: '0.65rem', opacity: 0.7 }}>Latency</div><div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{metrics.avg_response_latency?.toFixed(1)}s</div></div>
              </div>
            </div>
          )}

          {/* Questions Section with Full Details */}
          <div className="rev-questions">
            <h4 style={{ marginBottom: '1rem', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <i className="fas fa-question-circle"></i> Question & Answer Review
            </h4>
            {renderDetailedQuestions()}
          </div>
        </div>
      </div>
    </div>
  );
}