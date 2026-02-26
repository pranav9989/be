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
    let details = { mastery: 0, attempts: 0, status: 'not_started' };
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

  // ========== MASTERY COLOR FUNCTIONS ==========
  const getMasteryColor = (level) => {
    if (level >= 0.7) return '#10B981'; // Strong - Green
    if (level >= 0.4) return '#F59E0B'; // Medium - Orange
    return '#EF4444'; // Weak - Red
  };

  const getMasteryStatus = (level) => {
    if (level >= 0.7) return 'strong';
    if (level < 0.4) return 'weak';
    return 'medium';
  };

  // ========== SUBTOPIC STATUS FUNCTIONS ==========
  const getSubtopicStatusDisplay = (status, mastery) => {
    if (status === 'mastered') return 'mastered';
    if (status === 'ongoing') return 'ongoing';
    if (status === 'not_started') return 'not_started';
    // Fallback to mastery-based if status not available
    return getMasteryStatus(mastery);
  };

  const getSubtopicStatusColor = (status, mastery) => {
    const displayStatus = getSubtopicStatusDisplay(status, mastery);
    if (displayStatus === 'mastered' || displayStatus === 'strong') return '#10B981';
    if (displayStatus === 'ongoing' || displayStatus === 'medium') return '#F59E0B';
    if (displayStatus === 'weak') return '#EF4444';
    return '#94A3B8'; // not_started - gray
  };

  const getSubtopicStatusIcon = (status, mastery) => {
    const displayStatus = getSubtopicStatusDisplay(status, mastery);
    if (displayStatus === 'mastered') return '✅';
    if (displayStatus === 'strong') return '💪';
    if (displayStatus === 'ongoing') return '📚';
    if (displayStatus === 'weak') return '⚠️';
    return '🆕'; // not_started
  };

  const getSubtopicStatusText = (status, mastery) => {
    const displayStatus = getSubtopicStatusDisplay(status, mastery);
    if (displayStatus === 'mastered') return 'Mastered (3+ Q, ≥70%)';
    if (displayStatus === 'strong') return 'Strong Subtopic';
    if (displayStatus === 'ongoing') return 'In Progress';
    if (displayStatus === 'weak') return 'Needs Practice';
    return 'Not Started';
  };

  // ========== CONCEPT STATUS FUNCTIONS ==========
  const getConceptStatus = (conceptData) => {
    if (!conceptData) return 'new';
    if (conceptData.attempts < 3) return 'new';
    if (conceptData.is_weak) return 'weak';
    if (conceptData.is_strong) return 'strong';
    return 'medium';
  };

  const getConceptStatusColor = (status) => {
    if (status === 'strong') return '#10B981';
    if (status === 'weak') return '#EF4444';
    if (status === 'medium') return '#F59E0B';
    return '#94A3B8'; // new
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
                  Avg Mastery: <strong style={{ color: getMasteryColor(progress.overall?.avg_mastery || 0) }}>
                    {formatMastery(progress.overall?.avg_mastery || 0)}
                  </strong>
                </span>
              </div>
            </div>

            {/* Score Explanation */}
            <div className="score-explanation">
              <p><strong>📊 True Scores (No Inflation):</strong></p>
              <ul>
                <li><span className="color-dot strong"></span> <strong>Strong Subtopics:</strong> Mastery ≥ 70%</li>
                <li><span className="color-dot medium"></span> <strong>In Progress:</strong> 40% ≤ Mastery &lt; 70%</li>
                <li><span className="color-dot weak"></span> <strong>Weak Subtopics:</strong> Mastery &lt; 40%</li>
                <li><span className="color-dot mastered"></span> <strong>Mastered:</strong> 3+ questions & mastery ≥ 70%</li>
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

            {/* Subtopics by Status */}
            {subtopicStats && (
              <>
                {/* Mastered Subtopics */}
                {subtopicStats.by_topic && Object.entries(subtopicStats.by_topic).map(([topicName, topicData]) => {
                  const mastered = Object.entries(topicData.subtopics || {}).filter(
                    ([_, data]) => data.status === 'mastered'
                  );

                  if (mastered.length === 0) return null;

                  return (
                    <div key={topicName} className="mastered-section">
                      <h4 className="section-header">
                        <i className="fas fa-check-circle" style={{ color: '#10B981' }}></i>
                        Mastered Subtopics - {topicName}
                      </h4>
                      <div className="subtopic-section">
                        {mastered.map(([name, data]) => (
                          <div
                            key={name}
                            className="topic-chip mastered"
                            onClick={() => handleSubtopicClick(topicName, name)}
                          >
                            <span><strong>{name}</strong></span>
                            <div>
                              <span className="chip-value">{formatMastery(data.mastery)}</span>
                              <span className="chip-count">{data.attempts} Q</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}

                {/* Weak Subtopics */}
                {subtopicStats.by_topic && Object.entries(subtopicStats.by_topic).map(([topicName, topicData]) => {
                  const weak = Object.entries(topicData.subtopics || {}).filter(
                    ([_, data]) => data.status === 'weak'
                  );

                  if (weak.length === 0) return null;

                  return (
                    <div key={topicName} className="weak-section">
                      <h4 className="section-header">
                        <i className="fas fa-exclamation-triangle" style={{ color: '#EF4444' }}></i>
                        Need Practice - {topicName}
                      </h4>
                      <div className="subtopic-section">
                        {weak.map(([name, data]) => (
                          <div
                            key={name}
                            className="topic-chip weak"
                            onClick={() => handleSubtopicClick(topicName, name)}
                          >
                            <span><strong>{name}</strong></span>
                            <div>
                              <span className="chip-value">{formatMastery(data.mastery)}</span>
                              <span className="chip-count">{data.attempts} Q</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}

                {/* Ongoing Subtopics */}
                {subtopicStats.by_topic && Object.entries(subtopicStats.by_topic).map(([topicName, topicData]) => {
                  const ongoing = Object.entries(topicData.subtopics || {}).filter(
                    ([_, data]) => data.status === 'ongoing'
                  );

                  if (ongoing.length === 0) return null;

                  return (
                    <div key={topicName} className="ongoing-section">
                      <h4 className="section-header">
                        <i className="fas fa-spinner" style={{ color: '#F59E0B' }}></i>
                        In Progress - {topicName}
                      </h4>
                      <div className="subtopic-section">
                        {ongoing.map(([name, data]) => (
                          <div
                            key={name}
                            className="topic-chip ongoing"
                            onClick={() => handleSubtopicClick(topicName, name)}
                          >
                            <span><strong>{name}</strong></span>
                            <div>
                              <span className="chip-value">{formatMastery(data.mastery)}</span>
                              <span className="chip-count">{data.attempts} Q</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}

                {/* Not Started Subtopics */}
                {subtopicStats.by_topic && Object.entries(subtopicStats.by_topic).map(([topicName, topicData]) => {
                  const notStarted = topicData.not_started || 0;
                  if (notStarted === 0) return null;

                  return (
                    <div key={topicName} className="not-started-section">
                      <h4 className="section-header">
                        <i className="fas fa-clock" style={{ color: '#94A3B8' }}></i>
                        Not Started - {topicName}
                      </h4>
                      <div className="subtopic-section">
                        <span className="not-started-count">{notStarted} subtopics available</span>
                      </div>
                    </div>
                  );
                })}
              </>
            )}

            {/* Topic Mastery Grid */}
            <h4 className="topics-title">Topic Mastery (Overview)</h4>
            <div className="topics-grid">
              {Object.entries(progress.topics || {}).map(([topic, data]) => (
                <div
                  key={topic}
                  className="topic-card minimal"
                  onClick={() => loadTopicDetails(topic)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="topic-name">
                    {topic}
                  </div>

                  <div className="topic-simple-stats">
                    <span className="topic-mastery" style={{ color: getMasteryColor(data.mastery_level) }}>
                      {formatMastery(data.mastery_level)}
                    </span>
                    <span className="topic-questions">
                      {data.questions_attempted || 0} questions
                    </span>
                  </div>

                  <button
                    className="reset-btn"
                    onClick={(e) => {
                      e.stopPropagation();
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

              <div
                className="topic-status-badge"
                style={{
                  background: getSubtopicStatusColor(subtopicDetails?.status, subtopicDetails?.mastery),
                  color: 'white'
                }}
              >
                {getSubtopicStatusIcon(subtopicDetails?.status, subtopicDetails?.mastery)} {getSubtopicStatusText(subtopicDetails?.status, subtopicDetails?.mastery)}
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
                    color: getSubtopicStatusColor(subtopicDetails?.status, subtopicDetails?.mastery)
                  }}>
                    {subtopicDetails?.status || 'not_started'}
                  </strong>
                </div>
              </div>

              {/* Concept Status Explanation */}
              <div className="concept-explanation" style={{ margin: '1rem 0', fontSize: '0.85rem', background: '#F1F5F9', padding: '0.75rem', borderRadius: '8px' }}>
                <p><strong>📊 Concept Status (after 3 attempts):</strong></p>
                <ul style={{ margin: '0.5rem 0 0 1.5rem' }}>
                  <li><span style={{ color: '#10B981' }}>Strong</span> = correct ratio {'>'} 70%</li>
                  <li><span style={{ color: '#EF4444' }}>Weak</span> = miss ratio {'>'} 70%</li>
                  <li><span style={{ color: '#F59E0B' }}>Medium</span> = between thresholds</li>
                  <li><span style={{ color: '#94A3B8' }}>New</span> = {'<'} 3 attempts</li>
                </ul>
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
                      {/* Show sampled concepts if available */}
                      {q.sampled_concepts && q.sampled_concepts.length > 0 && (
                        <div className="sampled-concepts">
                          <strong>Concepts asked:</strong> {q.sampled_concepts.join(', ')}
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

              <div className="mastery-details">
                <div className="detail-item">
                  <span>Mastery Level</span>
                  <strong style={{ color: getMasteryColor(topicDetails.mastery?.mastery_level || 0) }}>
                    {formatMastery(topicDetails.mastery?.mastery_level || 0)}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>Current Difficulty</span>
                  <strong className={`difficulty-${topicDetails.mastery?.current_difficulty || 'medium'}`}>
                    {topicDetails.mastery?.current_difficulty || 'medium'}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>Questions Attempted</span>
                  <strong>{topicDetails.mastery?.questions_attempted || 0}</strong>
                </div>
                <div className="detail-item">
                  <span>Learning Velocity</span>
                  <strong style={{ color: (topicDetails.mastery?.learning_velocity || 0) > 0 ? '#10B981' : '#EF4444' }}>
                    {((topicDetails.mastery?.learning_velocity || 0) * 100).toFixed(1)}%
                  </strong>
                </div>
              </div>

              {/* Concept-Level Data */}
              {topicDetails.mastery?.concept_masteries && (
                <div className="concept-masteries">
                  <h4>📚 Concept Mastery (True scores)</h4>
                  <div className="concept-grid">
                    {Object.entries(topicDetails.mastery.concept_masteries).map(([conceptName, data]) => {
                      const status = getConceptStatus(data);
                      return (
                        <div key={conceptName} className="concept-item">
                          <div className="concept-header">
                            <span className="concept-name">{conceptName}</span>
                            <span className="concept-status" style={{
                              background: getConceptStatusColor(status),
                              color: 'white',
                              padding: '2px 8px',
                              borderRadius: '12px',
                              fontSize: '0.7rem'
                            }}>
                              {status}
                            </span>
                          </div>
                          <div className="concept-stats">
                            <div>Attempts: {data.attempts || 0}</div>
                            <div>Mentioned: {data.times_mentioned || 0}</div>
                            <div>Missed: {data.times_missed_when_sampled || 0}</div>
                            <div>Mastery: {((data.mastery_level || 0) * 100).toFixed(1)}%</div>
                            {data.stagnation_count > 0 && (
                              <div className="stagnation">Stagnation: {data.stagnation_count}</div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Recent Questions */}
              {topicDetails.recent_questions?.length > 0 && (
                <div className="recent-questions">
                  <h4>Recent Questions</h4>
                  {topicDetails.recent_questions.map((q, idx) => (
                    <div key={idx} className="recent-question">
                      <p className="question-text"><strong>Q:</strong> {q.question}</p>
                      {q.answer && <p className="answer-text"><strong>A:</strong> {q.answer}</p>}
                      {q.expected_answer && (
                        <div className="expected-answer">
                          <strong>Expected:</strong> {q.expected_answer}
                        </div>
                      )}
                      {/* Show sampled concepts */}
                      {q.sampled_concepts && q.sampled_concepts.length > 0 && (
                        <div className="sampled-concepts">
                          <strong>Concepts asked:</strong> {q.sampled_concepts.join(', ')}
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
                    setSelectedTopic(null);
                    setResetTopic(selectedTopic);
                    setShowResetConfirm(true);
                  }}
                  style={{ background: '#FEE2E2', color: '#991B1B' }}
                >
                  <i className="fas fa-redo-alt"></i> Reset {selectedTopic} Mastery
                </button>
              </div>
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