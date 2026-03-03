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
    // Try different locations where metrics might be stored
    let metrics = null;

    // Direct metrics field
    if (session.metrics) {
      metrics = session.metrics;
    }
    // Inside data.metrics
    else if (session.data?.metrics) {
      metrics = session.data.metrics;
    }
    // speech_metrics JSON string
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

                    {/* 🔥 SESSION SCORE - from session.score or metrics */}
                    {session.score ? (
                      <div className="ih-stats">
                        <span className="ih-score" style={{
                          background: session.score >= 70 ? 'rgba(16, 185, 129, 0.2)' :
                            session.score >= 40 ? 'rgba(245, 158, 11, 0.2)' :
                              'rgba(239, 68, 68, 0.2)',
                          color: session.score >= 70 ? '#10B981' :
                            session.score >= 40 ? '#F59E0B' :
                              '#EF4444'
                        }}>
                          Score: {Math.round(session.score)}/100
                        </span>
                        {session.duration && (
                          <span className="ih-duration"><i className="far fa-clock"></i> {Math.round(session.duration / 60)} min</span>
                        )}
                      </div>
                    ) : metrics?.avg_semantic_similarity ? (
                      <div className="ih-stats">
                        <span className="ih-score" style={{
                          background: metrics.avg_semantic_similarity >= 0.7 ? 'rgba(16, 185, 129, 0.2)' :
                            metrics.avg_semantic_similarity >= 0.4 ? 'rgba(245, 158, 11, 0.2)' :
                              'rgba(239, 68, 68, 0.2)',
                          color: metrics.avg_semantic_similarity >= 0.7 ? '#10B981' :
                            metrics.avg_semantic_similarity >= 0.4 ? '#F59E0B' :
                              '#EF4444'
                        }}>
                          Score: {Math.round(metrics.avg_semantic_similarity * 100)}/100
                        </span>
                      </div>
                    ) : null}

                    {/* 🔥 SESSION METRICS PREVIEW - wpm, speaking time, etc. */}
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

// ─── Modal to display ALL session metrics ───────────────────────────────────

function SessionModal({ session, onClose, getSessionName, formatDate }) {
  const data = session.data || {};
  const questions = data.questions || [];
  const answers = data.answers || {};

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
  const score = session.score || (metrics?.avg_semantic_similarity ? Math.round(metrics.avg_semantic_similarity * 100) : null);

  // 🔥 Format duration
  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  };

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
          <h4 className="rev-q">Q{i + 1}: {q.question}</h4>
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
          <h4 className="rev-q">Challenge {i + 1}: {c.topic} ({c.language.toUpperCase()})</h4>
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

          {/* 🔥 SESSION METRICS SUMMARY - ALL METRICS FROM YOUR LOGS */}
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
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Effective Duration</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{formatDuration(metrics.effective_duration)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Forced Silence</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{formatDuration(metrics.forced_silence_time)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Questions Answered</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{metrics.questions_answered || 0}</div>
                </div>

                {/* Row 2 */}
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Speaking Time</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#6ee7b7' }}>{formatDuration(metrics.speaking_time)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Silence (Thinking)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#f472b6' }}>{formatDuration(metrics.silence_time)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Speaking Ratio</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{Math.round(metrics.speaking_ratio * 100)}%</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Total Words</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{metrics.total_words || 0}</div>
                </div>

                {/* Row 3 */}
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Words Per Minute</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{Math.round(metrics.wpm)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Long Pauses</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{metrics.long_pause_count || 0}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Response Latency</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{metrics.avg_response_latency?.toFixed(2)}s</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Fluency Score</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{Math.round(metrics.fluency_score * 100)}%</div>
                </div>

                {/* Row 4 - Semantic & Keyword */}
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Avg Semantic</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: metrics.avg_semantic_similarity >= 0.7 ? '#10B981' : metrics.avg_semantic_similarity >= 0.4 ? '#F59E0B' : '#EF4444' }}>
                    {Math.round(metrics.avg_semantic_similarity * 100)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Avg Keyword</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: metrics.avg_keyword_coverage >= 0.6 ? '#10B981' : '#F59E0B' }}>
                    {Math.round(metrics.avg_keyword_coverage * 100)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Overall Relevance</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
                    {Math.round((metrics.avg_semantic_similarity * 0.8 + metrics.avg_keyword_coverage * 0.2) * 100)}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', opacity: 0.7 }}>Articulation Rate</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{metrics.articulation_rate?.toFixed(2)} w/s</div>
                </div>
              </div>
            </div>
          )}

          <div className="rev-summary">
            {score !== null && (
              <span>Overall Score: <strong style={{
                color: score >= 70 ? '#10B981' : score >= 40 ? '#F59E0B' : '#EF4444'
              }}>{Math.round(score)}/100</strong></span>
            )}
          </div>

          <div className="rev-questions">
            {renderedQuestions.length > 0 ? renderedQuestions : <p>No detailed data saved for this session.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}