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
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [topicDetails, setTopicDetails] = useState(null);
  const [message, setMessage] = useState({ type: '', text: '' });

  useEffect(() => {
    loadStats();
    loadProgress();
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

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };

  const getMasteryColor = (level) => {
    if (level >= 0.7) return '#10B981';
    if (level >= 0.4) return '#F59E0B';
    return '#EF4444';
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
            {message.text}
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

        {/* Learning Progress Section - ADAPTIVE LEARNING */}
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

            {/* Strengths & Weaknesses */}
            <div className="strength-weakness-grid">
              <div className="strength-card">
                <h4><i className="fas fa-trophy"></i> Your Strengths</h4>
                {progress.overall.strongest_topics.length > 0 ? (
                  progress.overall.strongest_topics.map(topic => {
                    const mastery = progress.topics[topic]?.mastery_level || 0;
                    return (
                      <div key={topic} className="topic-chip strong">
                        <span>{topic}</span>
                        <span className="chip-value">{formatMastery(mastery)}</span>
                      </div>
                    );
                  })
                ) : (
                  <p className="no-data">No strong topics identified yet</p>
                )}
              </div>
              <div className="weakness-card">
                <h4><i className="fas fa-exclamation-triangle"></i> Areas to Improve</h4>
                {progress.overall.weakest_topics.length > 0 ? (
                  progress.overall.weakest_topics.map(topic => {
                    const mastery = progress.topics[topic]?.mastery_level || 0;
                    return (
                      <div key={topic} className="topic-chip weak">
                        <span>{topic}</span>
                        <span className="chip-value">{formatMastery(mastery)}</span>
                      </div>
                    );
                  })
                ) : (
                  <p className="no-data">No weak topics identified yet</p>
                )}
              </div>
            </div>

            {/* Topic Mastery Grid */}
            <h4 className="topics-title">Topic Mastery</h4>
            <div className="topics-grid">
              {Object.entries(progress.topics).map(([topic, data]) => (
                <div
                  key={topic}
                  className="topic-card"
                  onClick={() => loadTopicDetails(topic)}
                >
                  <div className="topic-header">
                    <span className="topic-name">{topic}</span>
                    <span className={`topic-difficulty ${data.current_difficulty}`}>
                      {data.current_difficulty}
                    </span>
                  </div>

                  <div className="mastery-bar">
                    <div
                      className="mastery-fill"
                      style={{
                        width: `${data.mastery_level * 100}%`,
                        backgroundColor: getMasteryColor(data.mastery_level)
                      }}
                    />
                  </div>

                  <div className="topic-stats">
                    <span><i className="fas fa-question-circle"></i> {data.questions_attempted}</span>
                    <span className="mastery-value">{formatMastery(data.mastery_level)}</span>
                  </div>

                  {data.learning_velocity !== 0 && (
                    <div className="velocity-badge"
                      style={{ color: data.learning_velocity > 0 ? '#10B981' : '#EF4444' }}>
                      <i className={`fas fa-arrow-${data.learning_velocity > 0 ? 'up' : 'down'}`}></i>
                      {Math.abs(data.learning_velocity * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
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
                  <strong style={{ color: getMasteryColor(topicDetails.mastery.mastery_level) }}>
                    {formatMastery(topicDetails.mastery.mastery_level)}
                  </strong>
                </div>
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

              {topicDetails.mastery.missing_concepts?.length > 0 && (
                <div className="missing-section">
                  <h4>📝 Concepts to Focus On</h4>
                  <div className="concept-list">
                    {topicDetails.mastery.missing_concepts.map(concept => (
                      <span key={concept} className="concept-tag">{concept}</span>
                    ))}
                  </div>
                </div>
              )}

              {topicDetails.recent_questions?.length > 0 && (
                <div className="recent-questions">
                  <h4>Recent Questions</h4>
                  {topicDetails.recent_questions.map((q, idx) => (
                    <div key={idx} className="recent-question">
                      <p className="question-text">{q.question}</p>
                      {q.expected_answer && (
                        <div className="expected-answer">
                          <strong>Expected:</strong> {q.expected_answer}
                        </div>
                      )}
                      <div className="question-scores">
                        <span className="score semantic">S: {(q.semantic_score * 100).toFixed(1)}%</span>
                        <span className="score keyword">K: {(q.keyword_score * 100).toFixed(1)}%</span>
                        <span className="score coverage">C: {(q.coverage_score * 100).toFixed(1)}%</span>
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