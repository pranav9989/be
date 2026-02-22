import React, { useState, useEffect } from 'react';
import Header from '../Layout/Header';
import SkillsManager from './SkillsManager';
import { profileAPI, statsAPI, progressAPI } from '../../services/api';
import './Profile.css';

const Profile = ({ user, onLogout }) => {
  const [profile, setProfile] = useState(user);
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ sessions: 0 });
  const [progress, setProgress] = useState(null);
  const [subtopicStats, setSubtopicStats] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [topicDetails, setTopicDetails] = useState(null);
  const [selectedSubtopic, setSelectedSubtopic] = useState(null);
  const [subtopicDetails, setSubtopicDetails] = useState(null);
  const [subtopicQuestions, setSubtopicQuestions] = useState([]);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetTopic, setResetTopic] = useState(null);

  useEffect(() => {
    loadStats();
    loadProgress();
    loadSubtopicStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await statsAPI.getUserStats();
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const loadProgress = async () => {
    try {
      const response = await progressAPI.getUserProgress();
      if (response.data?.success) {
        setProgress(response.data.data);
      }
    } catch (error) {
      console.error('Failed to load progress:', error);
    }
  };

  const loadSubtopicStats = async () => {
    try {
      const response = await progressAPI.getSubtopicStats();
      if (response.data?.success) {
        setSubtopicStats(response.data.data);
      }
    } catch (error) {
      console.error('Failed to load subtopic stats:', error);
    }
  };

  const loadTopicDetails = async (topic) => {
    try {
      const response = await progressAPI.getTopicDetails(topic);
      if (response.data?.success) {
        setTopicDetails(response.data.data);
        setSelectedTopic(topic);
      }
    } catch (error) {
      console.error(`Failed to load ${topic} details:`, error);
    }
  };

  const loadSubtopicQuestions = async (topic, subtopic) => {
    try {
      const response = await progressAPI.getSubtopicQuestions(topic, subtopic);
      if (response.data?.success) {
        setSubtopicQuestions(response.data.questions);
      }
    } catch (error) {
      console.error(`Failed to load questions for ${subtopic}:`, error);
      setSubtopicQuestions([]);
    }
  };

  const handleSubtopicClick = async (topic, subtopicName) => {
    setSelectedSubtopic({ topic, name: subtopicName });

    // Get subtopic details from subtopicStats
    let details = { mastery: 0, attempts: 0, status: 'new' };
    if (subtopicStats?.by_topic?.[topic]?.subtopics?.[subtopicName]) {
      details = subtopicStats.by_topic[topic].subtopics[subtopicName];
    }
    setSubtopicDetails(details);

    // Load questions for this subtopic
    await loadSubtopicQuestions(topic, subtopicName);
  };

  const handleSave = async (formData) => {
    setLoading(true);
    try {
      await profileAPI.update(formData);
      setProfile(prev => ({ ...prev, ...formData }));
      setEditing(false);
      showMessage('success', 'Profile updated successfully!');
    } catch (error) {
      showMessage('error', 'Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const handleResetMastery = async (topic = null) => {
    try {
      const response = await progressAPI.resetMastery(topic);
      if (response.data?.success) {
        showMessage('success', response.data.message);
        // Reload all data
        await loadProgress();
        await loadSubtopicStats();
        setShowResetConfirm(false);
        setResetTopic(null);
      }
    } catch (error) {
      showMessage('error', 'Failed to reset mastery');
    }
  };

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };

  const getMasteryColor = (level) => {
    if (level >= 0.7) return '#10B981'; // Strong - Green
    if (level >= 0.4) return '#F59E0B'; // Medium - Orange
    return '#EF4444'; // Weak - Red
  };

  const getMasteryStatus = (level, stability) => {
    if (level >= 0.7 && stability >= 0.6) return 'strong';
    if (level < 0.4 || stability < 0.3) return 'weak';
    return 'medium';
  };

  const getSubtopicStatus = (mastery, status) => {
    if (status === 'weak') return 'weak';
    if (status === 'strong') return 'strong';
    if (mastery >= 0.7) return 'strong';
    if (mastery >= 0.4) return 'medium';
    return 'weak';
  };

  const formatMastery = (level) => {
    return `${(level * 100).toFixed(1)}%`;
  };

  return (
    <div className="profile-container">
      <Header
        user={user}
        onLogout={onLogout}
        title="Profile Management"
        showBack={true}
      />

      <main className="profile-main">
        {message.text && (
          <div className={`message ${message.type}`}>
            <i className={`fas ${message.type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}`}></i>
            {message.text}
          </div>
        )}

        {/* Reset Mastery Confirmation Modal */}
        {showResetConfirm && (
          <div className="topic-details-modal" onClick={() => setShowResetConfirm(false)}>
            <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '400px' }}>
              <h3>Confirm Reset</h3>
              <p style={{ margin: '1.5rem 0', lineHeight: '1.6' }}>
                Are you sure you want to reset {resetTopic ? `all mastery for ${resetTopic}` : 'ALL subtopic mastery'}?
                <br />
                <strong style={{ color: '#EF4444' }}>This action cannot be undone!</strong>
              </p>
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                <button
                  className="btn btn-primary"
                  onClick={() => handleResetMastery(resetTopic)}
                  style={{ background: '#EF4444' }}
                >
                  Yes, Reset
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => setShowResetConfirm(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        <section className="profile-header">
          <div className="profile-avatar">
            <i className="fas fa-user"></i>
          </div>
          <h2>{profile.full_name || profile.username}</h2>
          <p>Member since {new Date(profile.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</p>
        </section>

        <div className="profile-content">
          {/* Personal Information Card */}
          <div className="profile-info-card">
            <div className="card-header">
              <h3><i className="fas fa-user"></i> Personal Information</h3>
              <button
                onClick={() => setEditing(!editing)}
                className="edit-btn"
              >
                <i className="fas fa-edit"></i> {editing ? 'Cancel' : 'Edit'}
              </button>
            </div>

            {!editing ? (
              <div className="info-display">
                <div className="info-item">
                  <span className="info-label">Full Name</span>
                  <span className="info-value">{profile.full_name || 'Not provided'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Email</span>
                  <span className="info-value">{profile.email}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Phone</span>
                  <span className="info-value">{profile.phone || 'Not provided'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Experience</span>
                  <span className="info-value">{profile.experience_years || 0} years</span>
                </div>
              </div>
            ) : (
              <ProfileForm
                profile={profile}
                onSave={handleSave}
                onCancel={() => setEditing(false)}
                loading={loading}
              />
            )}
          </div>

          {/* Skills Card */}
          <SkillsManager
            skills={profile.skills || []}
            onSkillsUpdate={(newSkills) => {
              setProfile(prev => ({ ...prev, skills: newSkills }));
              handleSave({ skills: newSkills });
            }}
          />
        </div>

        {/* Learning Progress Section */}
        {progress && (
          <div className="progress-section">
            <div className="progress-header">
              <h3><i className="fas fa-chart-line"></i> Learning Progress</h3>
              <div className="progress-summary">
                <span className="avg-mastery">
                  Avg Mastery: <strong style={{ color: getMasteryColor(progress.overall.avg_mastery) }}>
                    {formatMastery(progress.overall.avg_mastery)}
                  </strong>
                </span>
              </div>
            </div>

            {/* Score Explanation */}
            <div className="score-explanation">
              <p><strong>📊 True Scores (No Inflation):</strong></p>
              <ul>
                <li><span className="color-dot strong"></span> <strong>Strong Subtopics:</strong> Mastery ≥ 70%</li>
                <li><span className="color-dot weak"></span> <strong>Weak Subtopics:</strong> Mastery &lt; 40%</li>
                <li><span className="color-dot medium"></span> <strong>In Progress:</strong> 40% ≤ Mastery &lt; 70%</li>
              </ul>
            </div>

            {/* Reset Mastery Button */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1.5rem' }}>
              <button
                className="btn btn-secondary"
                onClick={() => {
                  setResetTopic(null);
                  setShowResetConfirm(true);
                }}
                style={{ background: '#FEE2E2', color: '#991B1B', border: '1px solid #FECACA' }}
              >
                <i className="fas fa-redo-alt"></i> Reset All Mastery (Start Over)
              </button>
            </div>

            {/* Strengths, Weaknesses & In Progress */}
            <div className="sections-grid">
              {/* Strengths Section */}
              <div className="strength-card">
                <h4 className="section-header">
                  <i className="fas fa-trophy" style={{ color: '#10B981' }}></i>
                  Your Strengths
                </h4>
                {progress.strengths && progress.strengths.length > 0 ? (
                  <div className="subtopic-section">
                    {progress.strengths.map((item, idx) => (
                      <div
                        key={idx}
                        className="topic-chip strong"
                        onClick={() => handleSubtopicClick(item.topic, item.subtopic)}
                      >
                        <span>{item.topic} → <strong>{item.subtopic}</strong></span>
                        <div>
                          <span className="chip-value">{(item.mastery * 100).toFixed(1)}%</span>
                          <span className="chip-count">{item.attempts} Q</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-data">No strong subtopics yet. Keep practicing!</p>
                )}
              </div>

              {/* Weaknesses Section */}
              <div className="weakness-card">
                <h4 className="section-header">
                  <i className="fas fa-exclamation-triangle" style={{ color: '#EF4444' }}></i>
                  Areas to Improve
                </h4>
                {progress.weak_subtopics && progress.weak_subtopics.length > 0 ? (
                  <div className="subtopic-section">
                    {progress.weak_subtopics.map((item, idx) => (
                      <div
                        key={idx}
                        className="topic-chip weak"
                        onClick={() => handleSubtopicClick(item.topic, item.subtopic)}
                      >
                        <span>{item.topic} → <strong>{item.subtopic}</strong></span>
                        <div>
                          <span className="chip-value">{(item.mastery * 100).toFixed(1)}%</span>
                          <span className="chip-count">{item.attempts} Q</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-data">No weak subtopics! Great job!</p>
                )}
              </div>
            </div>

            {/* In Progress Section */}
            {progress.medium_subtopics && progress.medium_subtopics.length > 0 && (
              <div className="inprogress-card" style={{ marginTop: '1.5rem' }}>
                <h4 className="section-header">
                  <i className="fas fa-spinner" style={{ color: '#F59E0B' }}></i>
                  In Progress (40% - 70%)
                </h4>
                <div className="subtopic-section">
                  {progress.medium_subtopics.map((item, idx) => (
                    <div
                      key={idx}
                      className="topic-chip medium"
                      onClick={() => handleSubtopicClick(item.topic, item.subtopic)}
                    >
                      <span>{item.topic} → <strong>{item.subtopic}</strong></span>
                      <div>
                        <span className="chip-value">{(item.mastery * 100).toFixed(1)}%</span>
                        <span className="chip-count">{item.attempts} Q</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* All Attempted Subtopics Section */}
            {progress.all_subtopics && progress.all_subtopics.length > 0 && (
              <div className="all-subtopics-section">
                <h4 style={{ marginTop: '2rem', color: '#1E3A8A' }}>
                  <i className="fas fa-list"></i> All Attempted Subtopics
                </h4>
                <div className="subtopic-grid" style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                  gap: '0.75rem',
                  marginTop: '1rem'
                }}>
                  {progress.all_subtopics.map((item, idx) => {
                    const masteryColor = getMasteryColor(item.mastery);
                    return (
                      <div
                        key={idx}
                        className="subtopic-card"
                        onClick={() => handleSubtopicClick(item.topic, item.subtopic)}
                        style={{
                          background: '#F8FAFC',
                          padding: '0.75rem',
                          borderRadius: '8px',
                          border: '1px solid #E2E8F0',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontWeight: 500, color: '#1E3A8A' }}>
                            {item.topic} → {item.subtopic}
                          </span>
                          <span style={{
                            fontWeight: 600,
                            color: masteryColor,
                            background: '#FFFFFF',
                            padding: '2px 8px',
                            borderRadius: '12px',
                            fontSize: '0.75rem'
                          }}>
                            {(item.mastery * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div style={{ fontSize: '0.7rem', color: '#64748B', marginTop: '4px' }}>
                          {item.attempts} question{item.attempts !== 1 ? 's' : ''}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Topic Mastery Grid */}
            <h4 className="topics-title">Topic Mastery (Overview)</h4>
            <div className="topics-grid">
              {Object.entries(progress.topics).map(([topic, data]) => (
                <div
                  key={topic}
                  className="topic-card minimal"
                >
                  <div className="topic-name">
                    {topic}
                  </div>

                  <div className="topic-simple-stats">
                    <span className="topic-mastery">
                      {formatMastery(data.mastery_level)}
                    </span>
                    <span className="topic-questions">
                      {data.questions_attempted || 0} questions
                    </span>
                  </div>

                  <button
                    className="reset-btn"
                    onClick={() => {
                      setResetTopic(topic);
                      setShowResetConfirm(true);
                    }}
                  >
                    Reset
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Subtopic Details Modal */}
        {selectedSubtopic && (
          <div className="topic-details-modal" onClick={() => setSelectedSubtopic(null)}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
              <button className="close-btn" onClick={() => setSelectedSubtopic(null)}>×</button>

              <h3>{selectedSubtopic.topic} - {selectedSubtopic.name}</h3>

              <div className={`topic-status-badge ${subtopicDetails?.status || 'medium'}`}>
                {subtopicDetails?.status === 'strong' && '💪 Strong Subtopic'}
                {subtopicDetails?.status === 'weak' && '⚠️ Weak Subtopic - Needs Practice'}
                {(!subtopicDetails?.status || subtopicDetails.status === 'medium') && '📚 Subtopic In Progress'}
                {subtopicDetails?.status === 'new' && '🆕 New Subtopic'}
              </div>

              <div className="mastery-details">
                <div className="detail-item">
                  <span>Mastery Level</span>
                  <strong style={{ color: getMasteryColor(subtopicDetails?.mastery || 0) }}>
                    {formatMastery(subtopicDetails?.mastery || 0)}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>Questions Attempted</span>
                  <strong>{subtopicDetails?.attempts || 0}</strong>
                </div>
                <div className="detail-item">
                  <span>Status</span>
                  <strong style={{
                    color: subtopicDetails?.status === 'strong' ? '#10B981' :
                      subtopicDetails?.status === 'weak' ? '#EF4444' : '#F59E0B'
                  }}>
                    {subtopicDetails?.status || 'medium'}
                  </strong>
                </div>
              </div>

              {/* Recent Questions for this Subtopic */}
              {subtopicQuestions.length > 0 && (
                <div className="recent-questions">
                  <h4>Recent Questions</h4>
                  {subtopicQuestions.map((q, idx) => (
                    <div key={idx} className="recent-question">
                      <p className="question-text"><strong>Q:</strong> {q.question}</p>
                      {q.answer && <p className="answer-text"><strong>A:</strong> {q.answer}</p>}
                      {q.expected_answer && (
                        <div className="expected-answer">
                          <strong>Expected:</strong> {q.expected_answer}
                        </div>
                      )}
                      <div className="question-scores">
                        <span className="score semantic">Semantic: {(q.semantic_score * 100).toFixed(1)}%</span>
                        <span className="score keyword">Keyword: {(q.keyword_score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Reset button for this topic */}
              <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setSelectedSubtopic(null);
                    setResetTopic(selectedSubtopic.topic);
                    setShowResetConfirm(true);
                  }}
                  style={{ background: '#FEE2E2', color: '#991B1B' }}
                >
                  <i className="fas fa-redo-alt"></i> Reset {selectedSubtopic.topic} Mastery
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Topic Details Modal */}
        {selectedTopic && topicDetails && (
          <div className="topic-details-modal" onClick={() => setSelectedTopic(null)}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
              <button className="close-btn" onClick={() => setSelectedTopic(null)}>×</button>

              <h3>{selectedTopic} - Detailed Analysis</h3>

              <div className={`topic-status-badge ${getMasteryStatus(
                topicDetails.mastery.mastery_level,
                topicDetails.mastery.stability || 0
              )}`}>
                {getMasteryStatus(topicDetails.mastery.mastery_level, topicDetails.mastery.stability || 0) === 'strong' && '💪 Strong Topic'}
                {getMasteryStatus(topicDetails.mastery.mastery_level, topicDetails.mastery.stability || 0) === 'weak' && '⚠️ Weak Topic - Needs Practice'}
                {getMasteryStatus(topicDetails.mastery.mastery_level, topicDetails.mastery.stability || 0) === 'medium' && '📚 Topic In Progress'}
              </div>

              <div className="mastery-details">
                <div className="detail-item">
                  <span>Mastery Level</span>
                  <strong style={{ color: getMasteryColor(topicDetails.mastery.mastery_level) }}>
                    {formatMastery(topicDetails.mastery.mastery_level)}
                  </strong>
                </div>
                {topicDetails.mastery.stability > 0 && (
                  <div className="detail-item">
                    <span>Stability</span>
                    <strong>{(topicDetails.mastery.stability * 100).toFixed(1)}%</strong>
                  </div>
                )}
                <div className="detail-item">
                  <span>Current Difficulty</span>
                  <strong className={`difficulty-${topicDetails.mastery.current_difficulty}`}>
                    {topicDetails.mastery.current_difficulty}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>Questions Attempted</span>
                  <strong>{topicDetails.mastery.questions_attempted}</strong>
                </div>
                <div className="detail-item">
                  <span>Learning Velocity</span>
                  <strong style={{ color: topicDetails.mastery.learning_velocity > 0 ? '#10B981' : '#EF4444' }}>
                    {(topicDetails.mastery.learning_velocity * 100).toFixed(1)}%
                  </strong>
                </div>
              </div>

              {/* Weak Concepts Section */}
              {topicDetails.mastery.weak_concepts?.length > 0 && (
                <div className="missing-section weak">
                  <h4>⚠️ Concepts to Improve</h4>
                  <div className="concept-list">
                    {topicDetails.mastery.weak_concepts.map(concept => (
                      <span key={concept} className="concept-tag weak">{concept}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Strong Concepts Section */}
              {topicDetails.mastery.strong_concepts?.length > 0 && (
                <div className="missing-section strong">
                  <h4>💪 Mastered Concepts</h4>
                  <div className="concept-list">
                    {topicDetails.mastery.strong_concepts.map(concept => (
                      <span key={concept} className="concept-tag strong">{concept}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Missing Concepts */}
              {topicDetails.mastery.missing_concepts?.length > 0 && (
                <div className="missing-section">
                  <h4>📝 Recently Missed</h4>
                  <div className="concept-list">
                    {topicDetails.mastery.missing_concepts.map(concept => (
                      <span key={concept} className="concept-tag">{concept}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Reset button for this topic */}
              <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setSelectedTopic(null);
                    setResetTopic(selectedTopic);
                    setShowResetConfirm(true);
                  }}
                  style={{ background: '#FEE2E2', color: '#991B1B' }}
                >
                  <i className="fas fa-redo-alt"></i> Reset {selectedTopic} Mastery
                </button>
              </div>

              {topicDetails.recent_questions?.length > 0 && (
                <div className="recent-questions">
                  <h4>Recent Questions (True Scores)</h4>
                  {topicDetails.recent_questions.map((q, idx) => (
                    <div key={idx} className="recent-question">
                      <p className="question-text"><strong>Q:</strong> {q.question}</p>
                      {q.answer && <p className="answer-text"><strong>A:</strong> {q.answer}</p>}
                      {q.expected_answer && (
                        <div className="expected-answer">
                          <strong>Expected:</strong> {q.expected_answer}
                        </div>
                      )}
                      <div className="question-scores">
                        <span className="score semantic">Semantic: {(q.semantic_score * 100).toFixed(1)}%</span>
                        <span className="score keyword">Keyword: {(q.keyword_score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Statistics */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-number">{stats.sessions}</div>
            <div className="stat-label">Practice Sessions</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{profile.resume_filename ? '✓' : '✗'}</div>
            <div className="stat-label">Resume Uploaded</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{profile.skills?.length || 0}</div>
            <div className="stat-label">Skills Listed</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{progress?.overall?.total_questions || 0}</div>
            <div className="stat-label">Questions Answered</div>
          </div>
        </div>
      </main>
    </div>
  );
};

const ProfileForm = ({ profile, onSave, onCancel, loading }) => {
  const [formData, setFormData] = useState({
    full_name: profile.full_name || '',
    phone: profile.phone || '',
    experience_years: profile.experience_years || 0
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="profile-form">
      <div className="form-group">
        <label htmlFor="full_name">Full Name</label>
        <input
          type="text"
          id="full_name"
          value={formData.full_name}
          onChange={(e) => setFormData(prev => ({ ...prev, full_name: e.target.value }))}
          className="form-input"
        />
      </div>

      <div className="form-group">
        <label htmlFor="phone">Phone Number</label>
        <input
          type="tel"
          id="phone"
          value={formData.phone}
          onChange={(e) => setFormData(prev => ({ ...prev, phone: e.target.value }))}
          className="form-input"
        />
      </div>

      <div className="form-group">
        <label htmlFor="experience_years">Years of Experience</label>
        <input
          type="number"
          id="experience_years"
          value={formData.experience_years}
          onChange={(e) => setFormData(prev => ({ ...prev, experience_years: parseInt(e.target.value) || 0 }))}
          min="0"
          max="50"
          className="form-input"
        />
      </div>

      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Saving...' : 'Save Changes'}
        </button>
        <button type="button" onClick={onCancel} className="btn btn-secondary">
          Cancel
        </button>
      </div>
    </form>
  );
};

export default Profile;