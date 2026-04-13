import React, { useState, useRef, useEffect } from 'react';
import Header from '../Layout/Header';
import { resumeAPI } from '../../services/api';
import { useResumeInterviewStreaming } from '../../hooks/useResumeInterviewStreaming';
import PitchGraph from '../AgenticInterview/PitchGraph';
import './ResumeUpload.css';
import './ResumeInterview.css';

// ─── Mini components ─────────────────────────────────────────────────────────────
const SkillTag = ({ skill, variant = '' }) => (
  <span className={`skill-tag ${variant}`}>{skill}</span>
);

const SectionCard = ({ icon, title, children, accent }) => (
  <div className="analysis-card" style={accent ? { '--card-accent': accent } : {}}>
    <h4>
      <i className={icon} />
      {title}
    </h4>
    {children}
  </div>
);

const ScoreRing = ({ pct, label, color, size = 80 }) => {
  const r = (size / 2) - 7;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return (
    <div className="score-ring-wrap" title={`${pct}% ${label}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="score-ring-svg">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="7" />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dasharray 1s cubic-bezier(.4,0,.2,1)' }}
        />
      </svg>
      <div className="score-ring-inner">
        <strong style={{ color }}>{pct}%</strong>
        <span>{label}</span>
      </div>
    </div>
  );
};

// ─── Conversational Interview Component - WITH PITCH GRAPH ─────────────────────────
const ResumeConversationalInterview = ({ user, jobDescription, onBack }) => {
  const messagesEndRef = useRef(null);

  const {
    isConnected,
    isRecording,
    liveTranscript,
    startRecording,
    stopRecording,
    status,
    error,
    timeRemaining,
    messages,
    currentTurn,
    isInterviewerSpeaking,
    interviewDone,
    analysis,
    submitAnswer,
    metrics,
    livePitch,
    pitchHistory,
    pitchTimestamps,
    stabilityHistory,
    liveWpm
  } = useResumeInterviewStreaming(user.id);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, liveTranscript]);

  useEffect(() => {
    if (isConnected) {
      const timer = setTimeout(() => {
        startRecording();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [isConnected, startRecording]);

  const getTimeInSeconds = (timeStr) => {
    if (!timeStr || timeStr === '00:00') return 0;
    const parts = timeStr.split(':');
    if (parts.length === 2) {
      return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    }
    return 1800;
  };

  const formatDisplayTime = () => {
    const seconds = getTimeInSeconds(timeRemaining);
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (interviewDone) {
    return (
      <div className="resume-interview-container">
        <div className="interview-complete-card">
          <div className="complete-icon">🎉</div>
          <h2>Interview Complete!</h2>
          <p>Thank you for completing the interview. Here's your performance summary:</p>

          {metrics && (
            <div className="metrics-summary-complete">

              {/* SPEAKING METRICS */}
              <div className="metric-group speaking">
                <h4>🎤 Speaking Metrics</h4>
                <div className="metrics-grid">
                  <div className="metric">
                    <span className="metric-label">Speaking Time</span>
                    <span className="metric-value">{metrics.speaking_time?.toFixed(1) || 0}s</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Total User Turn Time</span>
                    <span className="metric-value">{metrics.total_user_turn_time?.toFixed(1) || 0}s</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Silence During Turn</span>
                    <span className="metric-value">{metrics.silence_during_turn?.toFixed(1) || 0}s</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Speaking Ratio</span>
                    <span className="metric-value" style={{
                      color: metrics.speaking_ratio >= 0.6 && metrics.speaking_ratio <= 0.75 ? '#4ade80' : '#f59e0b'
                    }}>{((metrics.speaking_ratio || 0) * 100).toFixed(1)}%</span>
                    <span className="metric-benchmark">(Ideal: 60-75%)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Session Duration</span>
                    <span className="metric-value">{metrics.session_duration?.toFixed(1) || 0}s</span>
                  </div>
                </div>
              </div>

              {/* FLUENCY METRICS */}
              <div className="metric-group fluency">
                <h4>⚡ Fluency Metrics</h4>
                <div className="metrics-grid">
                  <div className="metric">
                    <span className="metric-label">Words Per Minute (WPM)</span>
                    <span className="metric-value" style={{
                      color: metrics.wpm >= 120 && metrics.wpm <= 150 ? '#4ade80' : metrics.wpm > 180 ? '#ef4444' : '#f59e0b'
                    }}>{metrics.wpm?.toFixed(1) || 0}</span>
                    <span className="metric-benchmark">(Ideal: 120-150)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Articulation Rate</span>
                    <span className="metric-value" style={{
                      color: metrics.articulation_rate >= 2.0 && metrics.articulation_rate <= 2.5 ? '#4ade80' : metrics.articulation_rate > 3.0 ? '#ef4444' : '#f59e0b'
                    }}>{metrics.articulation_rate?.toFixed(2) || 0} words/s</span>
                    <span className="metric-benchmark">(Ideal: 2.0-2.5)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Avg Response Latency</span>
                    <span className="metric-value" style={{
                      color: metrics.avg_response_latency >= 1 && metrics.avg_response_latency <= 3 ? '#4ade80' : '#f59e0b'
                    }}>{metrics.avg_response_latency?.toFixed(2) || 0}s</span>
                    <span className="metric-benchmark">(Ideal: 1-3s)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Avg Pause Duration</span>
                    <span className="metric-value">{metrics.avg_pause_duration?.toFixed(2) || 0}s</span>
                    <span className="metric-benchmark">(Ideal: 1-3s)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Pause Count</span>
                    <span className="metric-value">{metrics.pause_count || 0}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Long Pauses (&gt;5s)</span>
                    <span className="metric-value" style={{
                      color: metrics.long_pause_count <= 2 ? '#4ade80' : '#ef4444'
                    }}>{metrics.long_pause_count || 0}</span>
                    <span className="metric-benchmark">(Ideal: &lt;2)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Hesitation Rate</span>
                    <span className="metric-value" style={{
                      color: metrics.hesitation_rate <= 10 ? '#4ade80' : metrics.hesitation_rate <= 15 ? '#f59e0b' : '#ef4444'
                    }}>{metrics.hesitation_rate?.toFixed(2) || 0}/min</span>
                    <span className="metric-benchmark">(Ideal: &lt;10/min)</span>
                  </div>
                </div>
              </div>

              {/* CONTENT QUALITY METRICS */}
              <div className="metric-group quality">
                <h4>📋 Content Quality</h4>
                <div className="metrics-grid">
                  <div className="metric">
                    <span className="metric-label">Semantic Similarity</span>
                    <span className="metric-value" style={{
                      color: metrics.avg_semantic_similarity >= 0.7 ? '#4ade80' : metrics.avg_semantic_similarity >= 0.5 ? '#f59e0b' : '#ef4444'
                    }}>{((metrics.avg_semantic_similarity || 0) * 100).toFixed(1)}%</span>
                    <span className="metric-benchmark">(Ideal: &lt;70%)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Keyword Coverage</span>
                    <span className="metric-value" style={{
                      color: metrics.avg_keyword_coverage >= 0.6 ? '#4ade80' : metrics.avg_keyword_coverage >= 0.4 ? '#f59e0b' : '#ef4444'
                    }}>{((metrics.avg_keyword_coverage || 0) * 100).toFixed(1)}%</span>
                    <span className="metric-benchmark">(Ideal: &lt;60%)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Overall Relevance</span>
                    <span className="metric-value" style={{
                      color: metrics.overall_relevance >= 0.7 ? '#4ade80' : metrics.overall_relevance >= 0.5 ? '#f59e0b' : '#ef4444'
                    }}>{((metrics.overall_relevance || 0) * 100).toFixed(1)}%</span>
                    <span className="metric-benchmark">(Ideal: &lt;70%)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Questions Answered</span>
                    <span className="metric-value">{metrics.questions_answered || 0}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Total Words</span>
                    <span className="metric-value">{metrics.total_words || 0}</span>
                  </div>
                </div>
              </div>

              {/* VOICE ANALYSIS METRICS */}
              <div className="metric-group voice">
                <h4>🎤 Voice Analysis</h4>
                <div className="metrics-grid">
                  <div className="metric">
                    <span className="metric-label">Average Pitch</span>
                    <span className="metric-value">{metrics.pitch_mean?.toFixed(1) || 0} Hz</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Pitch Range</span>
                    <span className="metric-value" style={{
                      color: metrics.pitch_range >= 50 && metrics.pitch_range <= 150 ? '#4ade80' : '#f59e0b'
                    }}>{metrics.pitch_range?.toFixed(1) || 0} Hz</span>
                    <span className="metric-benchmark">(Ideal: 50-150)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Pitch Stability</span>
                    <span className="metric-value" style={{
                      color: metrics.pitch_stability >= 70 ? '#4ade80' : metrics.pitch_stability >= 50 ? '#f59e0b' : '#ef4444'
                    }}>{metrics.pitch_stability?.toFixed(1) || 0}%</span>
                    <span className="metric-benchmark">(Ideal: &lt;70%)</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Pitch Variation (σ)</span>
                    <span className="metric-value">{metrics.pitch_std?.toFixed(1) || 0} Hz</span>
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* 🔥 NEW: Structured QA Review Cards */}
          {analysis?.qa_pairs && analysis.qa_pairs.length > 0 && (
            <div className="qa-review-section">
              <h3>📝 Question & Answer Review</h3>
              {analysis.qa_pairs.map((qa, index) => (
                <div key={index} className="qa-card">
                  <div className="qa-question">
                    <strong>Q{index + 1}:</strong> {qa.question}
                  </div>
                  <div className="qa-answer user">
                    <span>🗣️ Your Answer:</span>
                    <p>{qa.answer}</p>
                  </div>
                  <div className="qa-answer gold">
                    <span>💡 Ideal Answer:</span>
                    <p>{qa.gold_answer || qa.expected_answer || "Not available"}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Coaching feedback from analysis */}
          {analysis?.coaching_feedback && (
            <div className="coaching-feedback">
              <h3>🎯 Personalized Feedback</h3>
              <div className="coaching-content">
                {analysis.coaching_feedback.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
            </div>
          )}

          <div className="interview-actions">
            <button className="btn-secondary" onClick={onBack}>Back to Analysis</button>
            <button className="btn-primary" onClick={() => window.location.reload()}>New Interview</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="resume-interview-container">
      <div className="interview-header">
        <button className="back-btn" onClick={onBack}>
          <i className="fas fa-arrow-left"></i> Exit Interview
        </button>
        <div className="timer">
          <i className="fas fa-clock"></i> {formatDisplayTime()}
        </div>
        <div className="status">
          <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
          {isConnected ? 'Connected' : 'Connecting...'}
        </div>
      </div>

      {error && (
        <div className="error-message">
          <i className="fas fa-exclamation-circle"></i> {error}
        </div>
      )}

      <div className="interview-avatar-section">
        <div className={`avatar-circle ${isInterviewerSpeaking ? 'speaking' : ''}`}>
          <i className="fas fa-robot"></i>
        </div>
        <div className="interviewer-info">
          <h3>AI Interviewer</h3>
          <p>Based on your resume</p>
          {isInterviewerSpeaking && (
            <div className="speaking-indicator">
              <span></span><span></span><span></span>
            </div>
          )}
          {currentTurn === 'USER' && !isInterviewerSpeaking && (
            <div className="listening-indicator">
              <i className="fas fa-microphone"></i> Listening...
            </div>
          )}
        </div>
      </div>

      {/* Pitch graph */}
      {livePitch && livePitch.mean > 0 && (
        <div className="pitch-section-resume" style={{ margin: '1rem 0' }}>
          <PitchGraph
            pitchHistory={pitchHistory || []}
            stabilityHistory={stabilityHistory || []}
            pitchTimestamps={pitchTimestamps || []}
            livePitch={livePitch}
          />
        </div>
      )}

      {/* WPM display */}
      {currentTurn === 'USER' && liveWpm > 0 && (
        <div className="wpm-indicator" style={{
          textAlign: 'center',
          fontSize: '0.85rem',
          color: liveWpm >= 120 && liveWpm <= 160 ? '#4ade80' : liveWpm > 180 ? '#ef4444' : '#f59e0b',
          background: 'rgba(0,0,0,0.3)',
          padding: '4px 12px',
          borderRadius: '20px',
          display: 'inline-block',
          margin: '0 auto 10px auto',
          width: 'fit-content'
        }}>
          <i className="fas fa-tachometer-alt"></i> Speaking Rate: {liveWpm} WPM
          {liveWpm < 100 && ' (Too slow)'}
          {liveWpm >= 100 && liveWpm < 120 && ' (Slightly slow)'}
          {liveWpm >= 120 && liveWpm <= 160 && ' (Optimal)'}
          {liveWpm > 160 && liveWpm <= 180 && ' (Slightly fast)'}
          {liveWpm > 180 && ' (Too fast)'}
        </div>
      )}

      <div className="conversation-container">
        <div className="messages-list">
          {messages.filter(msg => msg.role !== 'gold').map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'interviewer' && <i className="fas fa-robot"></i>}
                {msg.role === 'user' && <i className="fas fa-user"></i>}
              </div>
              <div className="message-bubble">
                <div className="message-text">{msg.text}</div>
              </div>
            </div>
          ))}
          {liveTranscript && currentTurn === 'USER' && (
            <div className="message user live">
              <div className="message-avatar"><i className="fas fa-user"></i></div>
              <div className="message-bubble">
                <div className="message-text">{liveTranscript}<span className="cursor">|</span></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="recording-controls">
        {isRecording && (
          <div className="recording-indicator">
            <div className="recording-pulse"></div>
            <span>Recording your answer...</span>
          </div>
        )}
        <div className="status-text">{status}</div>

        {currentTurn === 'USER' && (
          <button
            className="submit-answer-btn"
            onClick={submitAnswer}
            disabled={currentTurn !== 'USER'}
          >
            <i className="fas fa-paper-plane"></i> Submit Answer
          </button>
        )}

        <button
          className={`end-interview-btn ${isRecording ? 'recording' : ''}`}
          onClick={stopRecording}
        >
          <i className="fas fa-stop-circle"></i> End Interview
        </button>
      </div>
    </div>
  );
};

// ─── Main Component ──────────────────────────────────────────────────────────────
const ResumeUpload = ({ user, onLogout }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [showInterview, setShowInterview] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [uploadedUser, setUploadedUser] = useState(null);

  const fileInputRef = useRef(null);

  const isValidFile = (file) => {
    const allowed = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    return allowed.includes(file.type) && file.size <= 16 * 1024 * 1024;
  };

  const handleFileSelect = (file) => {
    if (!file) return;
    if (!isValidFile(file)) {
      setError('Please select a valid PDF, DOC, or DOCX file (max 16MB)');
      return;
    }
    setSelectedFile(file);
    setError('');
    setResults(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) { setError('Please select a resume file'); return; }
    if (!jobDescription.trim()) { setError('Job Description is required to generate targeted questions'); return; }

    const formData = new FormData();
    formData.append('resume', selectedFile);
    formData.append('job_description', jobDescription.trim());

    setError('');
    setUploading(true);
    setUploadProgress(10);

    const progressInterval = setInterval(() => {
      setUploadProgress(p => p < 85 ? p + Math.random() * 12 : p);
    }, 400);

    try {
      const response = await resumeAPI.upload(formData);
      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.data.success) {
        throw new Error(response.data.message || 'Upload failed');
      }

      setUploadedUser({
        ...user,
        resume_filename: `${user.id}_${selectedFile.name}`,
        skills: response.data.data?.skills || [],
        experience_years: response.data.data?.experience_years || 0,
      });

      const analysisData = {
        ...response.data.data,
        job_fit_analysis: response.data.job_fit_analysis || response.data.data?.job_fit_analysis,
      };
      setResults(analysisData);

      setTimeout(() => setUploadProgress(0), 1500);
    } catch (err) {
      clearInterval(progressInterval);
      setUploadProgress(0);
      setError(err.response?.data?.message || err.message || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleStartConversationalInterview = () => {
    if (!results) {
      setError('Please upload your resume and add a job description first');
      return;
    }
    setShowInterview(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const effectiveUser = uploadedUser || user;

  if (showInterview) {
    return (
      <ResumeConversationalInterview
        user={effectiveUser}
        jobDescription={jobDescription}
        onBack={() => setShowInterview(false)}
      />
    );
  }

  return (
    <div className="resume-upload-container">
      <Header user={user} onLogout={onLogout} title="Resume Analysis" showBack={true} />

      <main className="resume-main">
        {!showInterview && (
          <>
            <div className="ru-page-hero">
              <div className="ru-hero-icon"><i className="fas fa-file-alt" /></div>
              <div className="ru-hero-text">
                <h1>Resume Analysis <span className="ru-hero-amp">&</span> Conversational Interview</h1>
                <p>Upload your resume + paste a job description to get AI-powered analysis and a realistic conversational interview experience.</p>
              </div>
            </div>

            <section className="upload-section">
              <h2><i className="fas fa-cloud-upload-alt" /> Upload Resume</h2>
              <p>PDF, DOC, or DOCX · Max 16 MB</p>

              <div
                className={`upload-area ${dragOver ? 'dragover' : ''} ${selectedFile ? 'has-file' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
              >
                {selectedFile ? (
                  <>
                    <div className="upload-file-icon">
                      <i className={selectedFile.name.endsWith('.pdf') ? 'fas fa-file-pdf' : 'fas fa-file-word'} />
                    </div>
                    <div className="upload-text">{selectedFile.name}</div>
                    <div className="upload-subtext">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB ·{' '}
                      <span className="upload-change">Click to change</span>
                    </div>
                  </>
                ) : (
                  <>
                    <i className="fas fa-arrow-up-from-bracket upload-icon" />
                    <div className="upload-text">Drag & drop or click to upload</div>
                    <div className="upload-subtext">PDF, DOC, DOCX up to 16 MB</div>
                  </>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                style={{ display: 'none' }}
                accept=".pdf,.doc,.docx"
                onChange={e => handleFileSelect(e.target.files?.[0])}
              />

              <div className="job-description-section">
                <label htmlFor="job-desc" className="job-description-label">
                  <i className="fas fa-briefcase" /> Job Description <span className="required-star">*</span>
                </label>
                <textarea
                  id="job-desc"
                  className="job-description-textarea"
                  value={jobDescription}
                  onChange={e => setJobDescription(e.target.value)}
                  placeholder="Paste the complete job description here. The AI will use this to perform skill-gap analysis and generate personalized interview questions..."
                  rows={5}
                />
                <small className="job-description-help">
                  <i className="fas fa-info-circle" /> More detailed JDs produce more accurate analysis and targeted questions.
                </small>
              </div>

              {uploadProgress > 0 && (
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
                </div>
              )}

              {error && (
                <div className="upload-error">
                  <i className="fas fa-exclamation-circle" /> {error}
                </div>
              )}

              <button
                className="upload-btn"
                onClick={handleUpload}
                disabled={!selectedFile || !jobDescription.trim() || uploading}
              >
                {uploading ? (
                  <><i className="fas fa-spinner fa-spin" /> Analysing Resume…</>
                ) : results ? (
                  <><i className="fas fa-redo" /> Re-Analyse</>
                ) : (
                  <><i className="fas fa-magic" /> Analyse Resume & Generate Insights</>
                )}
              </button>
            </section>

            {results && (
              <section className="results-section" id="analysis-results">
                <div className="results-header">
                  <div className="results-header-left">
                    <h3><i className="fas fa-chart-line" /> Analysis Complete</h3>
                    <p>{selectedFile?.name}</p>
                  </div>
                  <button className="start-mock-btn-inline" onClick={handleStartConversationalInterview}>
                    <i className="fas fa-play" /> Start Conversational Interview
                  </button>
                </div>

                <div className="analysis-grid">
                  {/* Resume Basics Card */}
                  <SectionCard icon="fas fa-id-card" title="Resume Basics">
                    <div className="analysis-item">
                      <h5><i className="fas fa-calendar-alt" /> Experience</h5>
                      <p className="experience-text">
                        {results.experience_years || 0} year{(results.experience_years || 0) !== 1 ? 's' : ''}
                        {results.experience_years === 0 && ' (fresher / not detected)'}
                      </p>
                    </div>

                    {results.skills?.length > 0 && (
                      <div className="analysis-item">
                        <h5><i className="fas fa-cogs" /> Skills ({results.skills.length})</h5>
                        <div className="skills-list">
                          {results.skills.map((s, i) => <SkillTag key={i} skill={s} />)}
                        </div>
                      </div>
                    )}
                  </SectionCard>

                  {/* Certifications Card */}
                  {results.certifications && results.certifications.length > 0 && (
                    <SectionCard icon="fas fa-certificate" title="Certifications & Courses">
                      <div className="analysis-item">
                        <div className="certifications-grid">
                          {results.certifications.slice(0, 10).map((cert, i) => (
                            <div key={i} className="certification-item">
                              <i className="fas fa-award"></i>
                              <span>{cert}</span>
                            </div>
                          ))}
                          {results.certifications.length > 10 && (
                            <div className="certification-more">+{results.certifications.length - 10} more</div>
                          )}
                        </div>
                      </div>
                    </SectionCard>
                  )}

                  {/* Projects Card */}
                  {results.projects && results.projects.length > 0 && (
                    <SectionCard icon="fas fa-project-diagram" title="Key Projects">
                      <div className="projects-grid">
                        {results.projects.map((project, idx) => {
                          let projectName = '';
                          let projectDesc = '';
                          let techStack = [];

                          if (typeof project === 'object') {
                            projectName = project.name || 'Project';
                            projectDesc = project.description || '';
                            techStack = project.tech_stack || [];
                          } else if (typeof project === 'string') {
                            const lines = project.split('\n');
                            projectName = lines[0] || 'Project';
                            projectDesc = project;
                          }

                          return (
                            <div key={idx} className="project-card">
                              <div className="project-header">
                                <i className="fas fa-code-branch"></i>
                                <h5>{projectName}</h5>
                              </div>
                              {techStack.length > 0 && (
                                <div className="project-tech-stack">
                                  {techStack.map((tech, tidx) => (
                                    <span key={tidx} className="tech-badge">{tech}</span>
                                  ))}
                                </div>
                              )}
                              <p className="project-description">
                                {projectDesc.slice(0, 200)}
                                {projectDesc.length > 200 ? '...' : ''}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </SectionCard>
                  )}

                  {/* Experience Card */}
                  {results.experience && results.experience.length > 0 && (
                    <SectionCard icon="fas fa-briefcase" title="Experience & Internships">
                      <div className="experience-list">
                        {results.experience.map((exp, idx) => {
                          let title = '';
                          let company = '';
                          let description = '';

                          if (typeof exp === 'object') {
                            title = exp.title || 'Position';
                            company = exp.company || '';
                            description = exp.description || '';
                          } else if (typeof exp === 'string') {
                            const lines = exp.split('\n');
                            title = lines[0] || 'Position';
                            description = exp;
                          }

                          return (
                            <div key={idx} className="experience-item">
                              <div className="experience-header">
                                <i className="fas fa-building"></i>
                                <div className="experience-title-group">
                                  <h5>{title}</h5>
                                  {company && <span className="experience-company">{company}</span>}
                                </div>
                              </div>
                              <p className="experience-description">
                                {description.slice(0, 200)}
                                {description.length > 200 ? '...' : ''}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </SectionCard>
                  )}

                  {/* Job Fit Analysis Card */}
                  {results.job_fit_analysis && (
                    <SectionCard icon="fas fa-bullseye" title="Job Fit Analysis">
                      <div className="match-score">
                        <ScoreRing
                          pct={Math.round(results.job_fit_analysis.match_percentage || 0)}
                          label="Match"
                          color={
                            results.job_fit_analysis.match_percentage >= 70 ? '#22c55e'
                              : results.job_fit_analysis.match_percentage >= 45 ? '#f59e0b'
                                : '#ef4444'
                          }
                        />
                        {results.job_fit_analysis.semantic_similarity > 0 && (
                          <ScoreRing
                            pct={Math.round(results.job_fit_analysis.semantic_similarity * 100)}
                            label="Semantic"
                            color="#9E95F5"
                          />
                        )}
                        <div className="fit-meta">
                          <span className={`gap-severity severity-${results.job_fit_analysis.gap_severity?.toLowerCase()}`}>
                            {results.job_fit_analysis.gap_severity} Skill Gap
                          </span>
                          <span className={`experience-fit ${results.job_fit_analysis.experience_fit === 'Good fit' ? 'good' : 'needs-work'}`}>
                            <i className="fas fa-user-clock" /> {results.job_fit_analysis.experience_fit}
                          </span>
                        </div>
                      </div>

                      {results.job_fit_analysis.matching_skills?.length > 0 && (
                        <div className="analysis-item">
                          <h5><i className="fas fa-check-circle" /> Matching Skills</h5>
                          <div className="skills-list">
                            {results.job_fit_analysis.matching_skills.map((s, i) => (
                              <SkillTag key={i} skill={s} variant="match" />
                            ))}
                          </div>
                        </div>
                      )}

                      {results.job_fit_analysis.missing_skills?.length > 0 && (
                        <div className="analysis-item">
                          <h5><i className="fas fa-exclamation-triangle" /> Skills to Develop</h5>
                          <div className="skills-list">
                            {results.job_fit_analysis.missing_skills.map((s, i) => (
                              <SkillTag key={i} skill={s} variant="missing" />
                            ))}
                          </div>
                        </div>
                      )}

                      {results.job_fit_analysis.jd_skills_found?.length > 0 && (
                        <div className="analysis-item">
                          <h5><i className="fas fa-search" /> Skills Found in JD</h5>
                          <div className="skills-list">
                            {results.job_fit_analysis.jd_skills_found.map((s, i) => (
                              <SkillTag key={i} skill={s} variant="" />
                            ))}
                          </div>
                        </div>
                      )}
                    </SectionCard>
                  )}
                </div>

                {/* START INTERVIEW CTA */}
                <div className="mock-interview-section">
                  <div className="interview-ready">
                    <div className="interview-ready-icon">
                      <i className="fas fa-robot" />
                    </div>
                    <div className="interview-ready-content">
                      <h4><i className="fas fa-star" /> Ready for Your Conversational Interview?</h4>
                      <p>
                        Our AI interviewer will ask{' '}
                        <strong>custom questions based on your resume and the job description</strong>.
                        This is a realistic conversational interview that lasts up to 30 minutes.
                      </p>
                      <div className="interview-features">
                        <span><i className="fas fa-brain" /> AI-Powered Questions</span>
                        <span><i className="fas fa-chart-bar" /> Real-time Analysis</span>
                        <span><i className="fas fa-microphone" /> Voice Recognition</span>
                        <span><i className="fas fa-clock" /> 30-Min Session</span>
                        <span><i className="fas fa-chart-line" /> Pitch Analysis</span>
                        <span><i className="fas fa-tachometer-alt" /> Real-time WPM</span>
                      </div>
                    </div>
                    <button className="mock-interview-btn" onClick={handleStartConversationalInterview}>
                      <i className="fas fa-play-circle" /> Start Conversational Interview
                    </button>
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
};

export default ResumeUpload;