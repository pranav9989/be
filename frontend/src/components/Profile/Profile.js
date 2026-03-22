import React, { useState, useEffect } from 'react';
import Header from '../Layout/Header';
import SkillsManager from './SkillsManager';
import InterviewHistory from './InterviewHistory';
import { profileAPI, statsAPI, progressAPI, interviewAPI } from '../../services/api';
import ActionPlanGenerator from '../ActionPlan/ActionPlanGenerator';
import './Profile.css';

const Profile = ({ user, onLogout }) => {
  const [profile, setProfile] = useState(user);
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ sessions: 0, metrics: null });
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
  const [loadingMetrics, setLoadingMetrics] = useState(true);

  // 🔥 Aggregated metrics from all sessions
  const [aggregatedMetrics, setAggregatedMetrics] = useState({
    avgSemantic: 0,
    avgKeyword: 0,
    avgWpm: 0,
    avgSpeakingTime: 0,
    avgSilenceTime: 0,
    avgSpeakingRatio: 0,
    avgResponseLatency: 0,
    avgArticulationRate: 0,
    totalSessions: 0,
    totalQuestions: 0,
    totalPracticeHours: 0
  });

  // Tabs
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadStats();
    loadProgress();
    loadSubtopicStats();
    loadAllSessionsMetrics();
  }, []);

  const loadStats = async () => {
    try {
      const response = await statsAPI.getUserStats();
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  // 🔥 Load all sessions and aggregate metrics
  // 🔥 Load all sessions and aggregate metrics
  const loadAllSessionsMetrics = async () => {
    setLoadingMetrics(true);
    try {
      const response = await interviewAPI.getUserHistory();
      if (response.data?.success) {
        const sessions = response.data.history || [];

        console.log('📊 Raw sessions data:', sessions); // Debug log

        let totalSemantic = 0;
        let totalKeyword = 0;
        let totalWpm = 0;
        let totalSpeakingTime = 0;
        let totalSilenceTime = 0;
        let totalSpeakingRatio = 0;
        let totalResponseLatency = 0;
        let totalArticulationRate = 0;
        let totalQuestions = 0;
        let sessionsWithMetrics = 0;

        sessions.forEach(session => {
          // Try to get metrics from different locations
          let sessionMetrics = null;

          if (session.metrics) {
            sessionMetrics = session.metrics;
          } else if (session.data?.metrics) {
            sessionMetrics = session.data.metrics;
          } else if (session.speech_metrics) {
            try {
              sessionMetrics = typeof session.speech_metrics === 'string'
                ? JSON.parse(session.speech_metrics)
                : session.speech_metrics;
            } catch (e) {
              console.log('Error parsing speech_metrics:', e);
            }
          }

          if (sessionMetrics) {
            console.log('📊 Session metrics found:', sessionMetrics);

            totalSemantic += sessionMetrics.avg_semantic_similarity || 0;
            totalKeyword += sessionMetrics.avg_keyword_coverage || 0;
            totalWpm += sessionMetrics.wpm || 0;
            totalSpeakingTime += sessionMetrics.speaking_time || 0;
            totalSilenceTime += sessionMetrics.silence_time || 0;
            totalSpeakingRatio += sessionMetrics.speaking_ratio || sessionMetrics.speaking_time_ratio || 0;
            totalResponseLatency += sessionMetrics.avg_response_latency || 0;
            totalArticulationRate += sessionMetrics.articulation_rate || 0;
            sessionsWithMetrics++;
          }

          // Count questions
          if (session.data?.qa_pairs && Array.isArray(session.data.qa_pairs)) {
            totalQuestions += session.data.qa_pairs.length;
          } else if (session.qa_pairs && Array.isArray(session.qa_pairs)) {
            totalQuestions += session.qa_pairs.length;
          }
        });

        if (sessionsWithMetrics > 0) {
          const avgSpeakingRatio = totalSpeakingRatio / sessionsWithMetrics;
          const avgSpeakingTime = totalSpeakingTime / sessionsWithMetrics;
          const avgSilenceTime = totalSilenceTime / sessionsWithMetrics;

          setAggregatedMetrics({
            avgSemantic: totalSemantic / sessionsWithMetrics,
            avgKeyword: totalKeyword / sessionsWithMetrics,
            avgWpm: totalWpm / sessionsWithMetrics,
            avgSpeakingTime: avgSpeakingTime,
            avgSilenceTime: avgSilenceTime,
            avgSpeakingRatio: avgSpeakingRatio,
            avgResponseLatency: totalResponseLatency / sessionsWithMetrics,
            avgArticulationRate: totalArticulationRate / sessionsWithMetrics,
            totalSessions: sessionsWithMetrics,
            totalQuestions: totalQuestions,
            totalPracticeHours: ((avgSpeakingTime + avgSilenceTime) * sessionsWithMetrics) / 3600,
          });

          console.log('📊 Final aggregated metrics:', {
            avgSemantic: totalSemantic / sessionsWithMetrics,
            avgKeyword: totalKeyword / sessionsWithMetrics,
            avgWpm: totalWpm / sessionsWithMetrics,
            sessionsWithMetrics
          });
        }
      }
    } catch (error) {
      console.error('Failed to load session metrics:', error);
    } finally {
      setLoadingMetrics(false);
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

    let details = { mastery: 0, attempts: 0, status: 'not_started' };
    if (subtopicStats?.by_topic?.[topic]?.subtopics?.[subtopicName]) {
      details = subtopicStats.by_topic[topic].subtopics[subtopicName];
    }
    setSubtopicDetails(details);
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

  // ========== UTILITY FUNCTIONS ==========
  const getMasteryColor = (level) => {
    if (level >= 0.7) return '#10B981';
    if (level >= 0.4) return '#F59E0B';
    return '#EF4444';
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatTime = (seconds) => {
    if (!seconds || seconds === 0) return '0s';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(0);
    return `${mins}m ${secs}s`;
  };

  const getOverallMastery = () => {
    if (!progress?.topics) return 0;
    const topics = Object.values(progress.topics);
    if (topics.length === 0) return 0;
    const sum = topics.reduce((acc, topic) => acc + (topic.mastery_level || 0), 0);
    return sum / topics.length;
  };

  // ========== GET METRICS ==========
  const metrics = {
    overallMastery: getOverallMastery(),
    avgSemantic: aggregatedMetrics.avgSemantic || 0,
    avgKeyword: aggregatedMetrics.avgKeyword || 0,
    avgWpm: aggregatedMetrics.avgWpm || 0,
    avgSpeakingTime: aggregatedMetrics.avgSpeakingTime || 0,
    avgSilenceTime: aggregatedMetrics.avgSilenceTime || 0,
    avgSpeakingRatio: aggregatedMetrics.avgSpeakingRatio || 0,
    avgResponseLatency: aggregatedMetrics.avgResponseLatency || 0,
    avgArticulationRate: aggregatedMetrics.avgArticulationRate || 0,
    totalSessions: aggregatedMetrics.totalSessions || 0,
    totalQuestions: aggregatedMetrics.totalQuestions || 0,
    totalPracticeHours: aggregatedMetrics.totalPracticeHours || 0
  };

  const getSubtopicStatusDisplay = (status, mastery) => {
    if (status === 'mastered') return 'mastered';
    if (status === 'ongoing') return 'ongoing';
    if (status === 'not_started') return 'not_started';
    return mastery >= 0.7 ? 'strong' : mastery < 0.4 ? 'weak' : 'medium';
  };

  const getSubtopicStatusColor = (status, mastery) => {
    const displayStatus = getSubtopicStatusDisplay(status, mastery);
    if (displayStatus === 'mastered' || displayStatus === 'strong') return '#10B981';
    if (displayStatus === 'ongoing' || displayStatus === 'medium') return '#F59E0B';
    if (displayStatus === 'weak') return '#EF4444';
    return '#94A3B8';
  };

  const getSubtopicStatusIcon = (status, mastery) => {
    const displayStatus = getSubtopicStatusDisplay(status, mastery);
    if (displayStatus === 'mastered') return '✅';
    if (displayStatus === 'strong') return '💪';
    if (displayStatus === 'ongoing') return '📚';
    if (displayStatus === 'weak') return '⚠️';
    return '🆕';
  };

  const getSubtopicStatusText = (status, mastery) => {
    const displayStatus = getSubtopicStatusDisplay(status, mastery);
    if (displayStatus === 'mastered') return 'Mastered';
    if (displayStatus === 'strong') return 'Strong';
    if (displayStatus === 'ongoing') return 'In Progress';
    if (displayStatus === 'weak') return 'Needs Practice';
    return 'Not Started';
  };

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
    return '#94A3B8';
  };

  const formatMastery = (level) => {
    return `${(level * 100).toFixed(1)}%`;
  };

  return (
    <div className="profile-container">
      <Header user={user} onLogout={onLogout} title="Profile Management" showBack={true} />

      <main className="profile-main">
        {message.text && (
          <div className={`message ${message.type}`}>
            <i className={`fas ${message.type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}`}></i>
            {message.text}
          </div>
        )}

        {/* Reset Mastery Modal */}
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
                <button className="btn btn-primary" onClick={() => handleResetMastery(resetTopic)} style={{ background: '#EF4444' }}>
                  Yes, Reset
                </button>
                <button className="btn btn-secondary" onClick={() => setShowResetConfirm(false)}>
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
        </section>

        <div className="profile-content">
          <div className="profile-info-card">
            <div className="card-header">
              <h3><i className="fas fa-user"></i> Personal Information</h3>
              <button onClick={() => setEditing(!editing)} className="edit-btn">
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
              <ProfileForm profile={profile} onSave={handleSave} onCancel={() => setEditing(false)} loading={loading} />
            )}
          </div>

          <SkillsManager
            skills={profile.skills || []}
            onSkillsUpdate={(newSkills) => {
              setProfile(prev => ({ ...prev, skills: newSkills }));
              handleSave({ skills: newSkills });
            }}
          />
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: '15px', marginBottom: '25px' }}>
          <button className={`btn ${activeTab === 'overview' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setActiveTab('overview')}>
            <i className="fas fa-chart-pie"></i> Performance Overview
          </button>
          <button className={`btn ${activeTab === 'history' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setActiveTab('history')}>
            <i className="fas fa-history"></i> Interview History
          </button>
          <button className={`btn ${activeTab === 'action_plan' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setActiveTab('action_plan')}>
            <i className="fas fa-magic"></i> Study Plan
          </button>
        </div>

        {activeTab === 'overview' && !editing && (
          <div className="overview-wrapper">
            {/* CLEAN PERFORMANCE SUMMARY - 3 COLUMNS */}
            <div className="performance-summary-card" style={{
              background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
              padding: '2rem',
              borderRadius: '20px',
              marginBottom: '2rem'
            }}>
              <h3 style={{ margin: '0 0 1.5rem 0', color: 'white', fontSize: '1.3rem' }}>
                <i className="fas fa-chart-line" style={{ marginRight: '10px' }}></i>
                Overall Performance
                {loadingMetrics && <span style={{ marginLeft: '10px', fontSize: '0.8rem', opacity: 0.7 }}>(Loading...)</span>}
              </h3>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '1.5rem'
              }}>
                {/* Column 1: Content Quality */}
                <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1.2rem', borderRadius: '12px' }}>
                  <h4 style={{ color: '#a78bfa', margin: '0 0 1rem 0', fontSize: '0.9rem', textTransform: 'uppercase' }}>
                    <i className="fas fa-brain" style={{ marginRight: '5px' }}></i> Content Quality
                  </h4>
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '0.2rem' }}>Overall Mastery</div>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: getMasteryColor(metrics.overallMastery) }}>
                      {formatPercentage(metrics.overallMastery)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#888' }}>Knowledge across topics</div>
                  </div>
                </div>

                {/* Column 2: Speech Metrics */}
                <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1.2rem', borderRadius: '12px' }}>
                  <h4 style={{ color: '#f472b6', margin: '0 0 1rem 0', fontSize: '0.9rem', textTransform: 'uppercase' }}>
                    <i className="fas fa-microphone-alt" style={{ marginRight: '5px' }}></i> Speech Metrics
                  </h4>
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '0.2rem' }}>Speaking Ratio</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: metrics.avgSpeakingRatio > 0.4 ? '#10B981' : '#F59E0B' }}>
                      {formatPercentage(metrics.avgSpeakingRatio)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#888' }}>Time speaking vs thinking</div>
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '0.2rem' }}>Words Per Minute</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: metrics.avgWpm > 100 && metrics.avgWpm < 180 ? '#10B981' : '#F59E0B' }}>
                      {Math.round(metrics.avgWpm)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#888' }}>Speech pace</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '0.2rem' }}>Articulation Rate</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'white' }}>
                      {metrics.avgArticulationRate.toFixed(2)} w/s
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#888' }}>Words per second</div>
                  </div>
                </div>

                {/* Column 3: Time & Volume */}
                <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1.2rem', borderRadius: '12px' }}>
                  <h4 style={{ color: '#6ee7b7', margin: '0 0 1rem 0', fontSize: '0.9rem', textTransform: 'uppercase' }}>
                    <i className="fas fa-clock" style={{ marginRight: '5px' }}></i> Time & Volume
                  </h4>
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '0.2rem' }}>Speaking Time</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'white' }}>
                      {formatTime(metrics.avgSpeakingTime)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#888' }}>Average per session</div>
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '0.2rem' }}>Silence (Thinking)</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'white' }}>
                      {formatTime(metrics.avgSilenceTime)}
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#888' }}>Pauses between speech</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '0.2rem' }}>Response Latency</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: metrics.avgResponseLatency < 2 ? '#10B981' : '#F59E0B' }}>
                      {metrics.avgResponseLatency.toFixed(2)}s
                    </div>
                    <div style={{ fontSize: '0.7rem', color: '#888' }}>Time to start answering</div>
                  </div>
                </div>
              </div>

              {/* Stats Footer */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: '1.5rem',
                paddingTop: '1.5rem',
                borderTop: '1px solid rgba(255,255,255,0.2)',
                color: 'white',
                fontSize: '0.9rem'
              }}>
                <span><strong>{metrics.totalSessions}</strong> Sessions</span>
                <span><strong>{metrics.totalQuestions}</strong> Questions</span>
                <span><strong>{metrics.totalPracticeHours.toFixed(1)}h</strong> Practice</span>
              </div>
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

                <div className="score-explanation">
                  <p><strong>📊 True Scores (No Inflation):</strong></p>
                  <ul>
                    <li><span className="color-dot strong"></span> <strong>Strong:</strong> ≥70%</li>
                    <li><span className="color-dot medium"></span> <strong>In Progress:</strong> 40-70%</li>
                    <li><span className="color-dot weak"></span> <strong>Weak:</strong> &lt;40%</li>
                    <li><span className="color-dot mastered"></span> <strong>Mastered:</strong> 3+ Q & ≥70%</li>
                  </ul>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1.5rem' }}>
                  <button className="btn btn-secondary" onClick={() => { setResetTopic(null); setShowResetConfirm(true); }}
                    style={{ background: '#FEE2E2', color: '#991B1B', border: '1px solid #FECACA' }}>
                    <i className="fas fa-redo-alt"></i> Reset All Mastery
                  </button>
                </div>

                <h4 className="topics-title">Topic Mastery</h4>
                <div className="topics-grid">
                  {Object.entries(progress.topics || {}).map(([topic, data]) => (
                    <div key={topic} className="topic-card minimal" onClick={() => loadTopicDetails(topic)} style={{ cursor: 'pointer' }}>
                      <div className="topic-name">{topic}</div>
                      <div className="topic-simple-stats">
                        <span className="topic-mastery" style={{ color: getMasteryColor(data.mastery_level) }}>
                          {formatMastery(data.mastery_level)}
                        </span>
                        <span className="topic-strength" style={{ color: getMasteryColor(data.mastery_level) }}>
                          {data.mastery_level >= 0.7 ? 'strong' : data.mastery_level < 0.4 ? 'weak' : 'medium'}
                        </span>
                        <span className="topic-questions">{data.questions_attempted || 0} Q</span>
                      </div>
                      <button className="reset-btn" onClick={(e) => { e.stopPropagation(); setResetTopic(topic); setShowResetConfirm(true); }}>
                        Reset
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Modals and Stats remain same... */}
            {/* Subtopic Details Modal */}
            {selectedSubtopic && (
              <div className="topic-details-modal" onClick={() => setSelectedSubtopic(null)}>
                <div className="modal-content" onClick={e => e.stopPropagation()}>
                  <button className="close-btn" onClick={() => setSelectedSubtopic(null)}>×</button>
                  <h3>{selectedSubtopic.topic} - {selectedSubtopic.name}</h3>
                  <div className="topic-status-badge" style={{ background: getSubtopicStatusColor(subtopicDetails?.status, subtopicDetails?.mastery), color: 'white' }}>
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
                      <strong style={{ color: getSubtopicStatusColor(subtopicDetails?.status, subtopicDetails?.mastery) }}>
                        {subtopicDetails?.status || 'not_started'}
                      </strong>
                    </div>
                  </div>
                  <div className="concept-explanation">
                    <p><strong>📊 Concept Status:</strong></p>
                    <ul>
                      <li><span style={{ color: '#10B981' }}>Strong</span> = correct &gt;70%</li>
                      <li><span style={{ color: '#EF4444' }}>Weak</span> = miss &gt;70%</li>
                      <li><span style={{ color: '#F59E0B' }}>Medium</span> = between</li>
                      <li><span style={{ color: '#94A3B8' }}>New</span> = &lt;3 attempts</li>
                    </ul>
                  </div>
                  {subtopicQuestions.length > 0 && (
                    <div className="recent-questions">
                      <h4>Recent Questions</h4>
                      {subtopicQuestions.map((q, idx) => (
                        <div key={idx} className="recent-question">
                          <p className="question-text"><strong>Q:</strong> {q.question}</p>
                          {q.answer && <p className="answer-text"><strong>A:</strong> {q.answer}</p>}
                          <div className="question-scores">
                            <span className="score semantic">Semantic: {(q.semantic_score * 100).toFixed(1)}%</span>
                            <span className="score keyword">Keyword: {(q.keyword_score * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
                    <button className="btn btn-secondary" onClick={() => { setSelectedSubtopic(null); setResetTopic(selectedSubtopic.topic); setShowResetConfirm(true); }}
                      style={{ background: '#FEE2E2', color: '#991B1B' }}>
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

                  {topicDetails.mastery?.concept_masteries && (
                    <div className="concept-masteries">
                      <h4>📚 Concept Mastery</h4>
                      <div className="concept-grid">
                        {Object.entries(topicDetails.mastery.concept_masteries).map(([conceptName, data]) => {
                          const status = getConceptStatus(data);
                          return (
                            <div key={conceptName} className="concept-item">
                              <div className="concept-header">
                                <span className="concept-name">{conceptName}</span>
                                <span className="concept-status" style={{ background: getConceptStatusColor(status), color: 'white', padding: '2px 8px', borderRadius: '12px', fontSize: '0.7rem' }}>
                                  {status}
                                </span>
                              </div>
                              <div className="concept-stats">
                                <div>Mastery: {((data.mastery_level || 0) * 100).toFixed(1)}%</div>
                                <div>Attempts: {data.attempts || 0}</div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {topicDetails.recent_questions?.length > 0 && (
                    <div className="recent-questions">
                      <h4>Recent Questions</h4>
                      {topicDetails.recent_questions.map((q, idx) => (
                        <div key={idx} className="recent-question">
                          <p className="question-text"><strong>Q:</strong> {q.question}</p>
                          <div className="question-scores">
                            <span className="score semantic">Semantic: {(q.semantic_score * 100).toFixed(1)}%</span>
                            <span className="score keyword">Keyword: {(q.keyword_score * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
                    <button className="btn btn-secondary" onClick={() => { setSelectedTopic(null); setResetTopic(selectedTopic); setShowResetConfirm(true); }}
                      style={{ background: '#FEE2E2', color: '#991B1B' }}>
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
          </div>
        )}

        {activeTab === 'history' && !editing && <InterviewHistory />}
        {activeTab === 'action_plan' && !editing && (
          <div style={{ marginTop: '20px' }}>
            <ActionPlanGenerator />
          </div>
        )}
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
        <input type="text" id="full_name" value={formData.full_name} onChange={(e) => setFormData(prev => ({ ...prev, full_name: e.target.value }))} className="form-input" />
      </div>
      <div className="form-group">
        <label htmlFor="phone">Phone Number</label>
        <input type="tel" id="phone" value={formData.phone} onChange={(e) => setFormData(prev => ({ ...prev, phone: e.target.value }))} className="form-input" />
      </div>
      <div className="form-group">
        <label htmlFor="experience_years">Years of Experience</label>
        <input type="number" id="experience_years" value={formData.experience_years} onChange={(e) => setFormData(prev => ({ ...prev, experience_years: parseInt(e.target.value) || 0 }))} min="0" max="50" className="form-input" />
      </div>
      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>{loading ? 'Saving...' : 'Save Changes'}</button>
        <button type="button" onClick={onCancel} className="btn btn-secondary">Cancel</button>
      </div>
    </form>
  );
};

export default Profile;