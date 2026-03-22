import React, { useState, useEffect, useRef } from 'react';
import Header from '../Layout/Header';
import { useInterviewStreaming } from '../../hooks/useInterviewStreaming';
import AvatarViewer from './AvatarViewer';
import { getPersona } from './interviewerPersonas';
import PitchGraph from './PitchGraph';
import './AgenticInterview.css';

// Session counter persisted to localStorage so it increments across page reloads
const getSessionCount = () => parseInt(localStorage.getItem('agenticSessionCount') || '0', 10);
const incrementSessionCount = () => {
    const next = getSessionCount() + 1;
    localStorage.setItem('agenticSessionCount', String(next));
    return next;
};

const AgenticInterview = ({ user, onLogout }) => {
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
        finalTranscript,
        metrics,
        sessionPlan,
        livePitch,
        pitchHistory,
        pitchTimestamps,
        stabilityHistory
    } = useInterviewStreaming(user.id);

    const [started, setStarted] = useState(false);
    const [showMetrics, setShowMetrics] = useState(false);
    const [expandedQA, setExpandedQA] = useState({});
    const [coachingFeedback, setCoachingFeedback] = useState(null);
    const [loadingCoaching, setLoadingCoaching] = useState(false);
    const [sessionId, setSessionId] = useState(null);
    const [persona] = useState(() => getPersona(getSessionCount()));
    const chatBoxRef = useRef(null);

    // Auto-scroll chat
    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    }, [messages, liveTranscript]);

    // Extract session ID from analysis when interview is done
    useEffect(() => {
        if (interviewDone && analysis && analysis.session_id) {
            setSessionId(analysis.session_id);
        }
    }, [interviewDone, analysis]);

    useEffect(() => {
        const fetchInterviewResults = async () => {
            if (sessionId && !coachingFeedback && !loadingCoaching) {
                setLoadingCoaching(true);
                try {
                    const token = localStorage.getItem('token');
                    // 🔥 FIX: Use full backend URL instead of relative path
                    const response = await fetch('http://localhost:5000/api/interview_results', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify({ session_id: sessionId })
                    });

                    const data = await response.json();
                    if (data.success) {
                        setCoachingFeedback(data.coaching_feedback);
                        if (data.metrics) {
                            console.log('Metrics from combined endpoint:', data.metrics);
                        }
                    } else {
                        console.error('Failed to fetch results:', data.error);
                    }
                } catch (err) {
                    console.error('Error fetching interview results:', err);
                } finally {
                    setLoadingCoaching(false);
                }
            }
        };

        fetchInterviewResults();
    }, [sessionId, coachingFeedback, loadingCoaching]);

    // Add this after your existing useEffects (around line 45)

    useEffect(() => {
        console.log('🔍 DEBUG: interviewDone =', interviewDone);
        console.log('🔍 DEBUG: analysis =', analysis);
        console.log('🔍 DEBUG: analysis.session_id =', analysis?.session_id);
        console.log('🔍 DEBUG: sessionId state =', sessionId);
    }, [interviewDone, analysis, sessionId]);

    const startAgenticInterview = async () => {
        setStarted(true);
        setShowMetrics(false);
        setCoachingFeedback(null);
        setSessionId(null);
        startRecording(false);
    };

    const handleEndInterview = () => {
        stopRecording();
        setShowMetrics(true);
    };

    const handleNewInterview = () => {
        incrementSessionCount();
        window.location.reload();
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const toggleQAExpanded = (index) => {
        setExpandedQA(prev => ({ ...prev, [index]: !prev[index] }));
    };

    // Helper function to safely get metrics values
    const getMetricValue = (key, defaultValue = 'N/A') => {
        if (metrics && metrics[key] !== undefined && metrics[key] !== null) {
            return metrics[key];
        }
        if (analysis && analysis.metrics && analysis.metrics[key] !== undefined) {
            return analysis.metrics[key];
        }
        if (analysis && analysis[key] !== undefined) {
            return analysis[key];
        }
        return defaultValue;
    };

    // ── Metrics panel with UNIFIED research-grade metrics ───────────────────
    const renderMetrics = () => {
        if (!analysis && !metrics) return null;

        const data = analysis || {};
        const metricsData = metrics || data.metrics || {};

        // Parse speech_metrics JSON if present
        let speechMetrics = {};
        if (data.speech_metrics) {
            try {
                speechMetrics = typeof data.speech_metrics === 'string'
                    ? JSON.parse(data.speech_metrics)
                    : data.speech_metrics;
            } catch (e) {
                console.log('Error parsing speech_metrics:', e);
            }
        }

        // Unified getValue function
        const getValue = (key, defaultValue = 'N/A') => {
            if (metricsData[key] !== undefined && metricsData[key] !== null) return metricsData[key];
            if (data.metrics && data.metrics[key] !== undefined) return data.metrics[key];
            if (data[key] !== undefined) return data[key];
            if (speechMetrics[key] !== undefined) return speechMetrics[key];
            return defaultValue;
        };

        // ✅ CORRECT UNIFIED METRICS KEYS (matching backend)
        const speakingTime = getValue('speaking_time');
        const totalUserTurnTime = getValue('total_user_turn_time');
        const forcedSilenceTime = getValue('forced_silence_time');
        const availableSpeakingTime = getValue('available_speaking_time');
        const silenceDuringTurn = getValue('silence_during_turn');
        const speakingRatio = getValue('speaking_ratio');
        const sessionDuration = getValue('session_duration');

        const wpm = getValue('wpm');
        const articulationRate = getValue('articulation_rate');
        const avgResponseLatency = getValue('avg_response_latency');
        const avgPauseDuration = getValue('avg_pause_duration');
        const pauseCount = getValue('pause_count');
        const longPauseCount = getValue('long_pause_count');
        const hesitationRate = getValue('hesitation_rate');

        // Content Quality - Using correct keys
        const semanticSimilarity = getValue('semantic_similarity') || 0;
        const keywordCoverage = getValue('keyword_coverage') || 0;
        const questionsAnswered = getValue('questions_answered');
        const totalWords = getValue('total_words');

        // Get overall relevance or calculate it
        let overallRelevance = getValue('overall_relevance');
        if ((overallRelevance === 'N/A' || overallRelevance === 0) &&
            semanticSimilarity !== 'N/A' && keywordCoverage !== 'N/A') {
            const semantic = typeof semanticSimilarity === 'number' ? semanticSimilarity : 0;
            const keyword = typeof keywordCoverage === 'number' ? keywordCoverage : 0;
            overallRelevance = (semantic * 0.8) + (keyword * 0.2);
        }

        // Voice Analysis
        const pitchMean = getValue('pitch_mean');
        const pitchStd = getValue('pitch_std');
        const pitchRange = getValue('pitch_range');
        const pitchStability = getValue('pitch_stability');

        const conversation = data.conversation ||
            (messages && messages.length > 0
                ? messages.map(m => `${m.role === 'interviewer' ? 'Interviewer' : 'You'}: ${m.text}`).join('\n\n')
                : finalTranscript || 'No transcript available');

        const qaPairs = data.qa_pairs || metricsData.question_scores || [];

        return (
            <div className="metrics-panel">
                {/* Mini persona badge */}
                <div className="metrics-persona-badge">
                    <div className="metrics-persona-avatar" style={{ background: `linear-gradient(135deg, ${persona.avatarColor}, ${persona.accentColor})` }}>
                        {persona.initials}
                    </div>
                    <div>
                        <p className="metrics-persona-name">Interviewed by {persona.name}</p>
                        <p className="metrics-persona-role">{persona.title} · {persona.company}</p>
                    </div>
                </div>

                {/* COACHING FEEDBACK SECTION - FIXED */}
                {(coachingFeedback || loadingCoaching) && (
                    <div className="coaching-section-wrapper">
                        {loadingCoaching ? (
                            <div className="coaching-loading">
                                <div className="spinner"></div>
                                <p>Generating personalized coaching feedback...</p>
                            </div>
                        ) : coachingFeedback ? (
                            <div className="coaching-feedback-container">
                                <h2>🎯 Personalized Coaching Feedback</h2>
                                <p className="coaching-intro">
                                    Based on your performance, here are specific, actionable recommendations to improve your interview skills.
                                </p>

                                {coachingFeedback.split(/\n(?=## )/).map((section, idx) => {
                                    const lines = section.split('\n');
                                    const title = lines[0];
                                    const content = lines.slice(1).join('\n');
                                    const iconMatch = title.match(/[🗣️📚⏱️🎯]/);
                                    const icon = iconMatch ? iconMatch[0] : '📌';

                                    return (
                                        <div key={idx} className="coaching-section">
                                            <h3 className="coaching-section-title">
                                                <span className="coaching-icon">{icon}</span>
                                                {title.replace(/^## /, '').replace(/[🗣️📚⏱️🎯]/g, '').trim()}
                                            </h3>
                                            <div className="coaching-section-content">
                                                {content.split('\n').map((line, i) => {
                                                    if (!line.trim()) return null;
                                                    if (line.trim().startsWith('-') || line.trim().startsWith('*')) {
                                                        return (
                                                            <div key={i} className="coaching-bullet">
                                                                <span className="bullet-marker">✓</span>
                                                                <span className="bullet-text">{line.replace(/^[-*]\s*/, '').trim()}</span>
                                                            </div>
                                                        );
                                                    }
                                                    return <p key={i} className="coaching-text">{line}</p>;
                                                })}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        ) : null}
                    </div>
                )}

                <h3>📊 Research-Grade Interview Analysis</h3>

                <div className="metrics-explanation">
                    <p><strong>All scores are raw, unaltered values based on research paper:</strong></p>
                    <ul>
                        <li><strong>Speaking Ratio</strong> = Speaking Time / Available Speaking Time (excluding forced waits)</li>
                        <li><strong>Semantic Similarity</strong> = True cosine similarity between your answer and expected answer</li>
                        <li><strong>Keyword Coverage</strong> = Percentage of technical terms you used (stop words filtered)</li>
                        <li><strong>Overall Relevance</strong> = 80% semantic + 20% keyword (weighted average per paper)</li>
                    </ul>
                </div>

                <div className="metrics-grid">
                    {/* Full Transcript */}
                    <div className="metric-section full-width">
                        <h4>📝 Complete Interview Transcript</h4>
                        <div className="transcript-box full-conversation">
                            {conversation.split('\n\n').map((line, i) => {
                                if (line.startsWith('Interviewer:')) return <div key={i} className="transcript-line interviewer-line">{line}</div>;
                                if (line.startsWith('User:')) return <div key={i} className="transcript-line user-line">{line}</div>;
                                return <div key={i} className="transcript-line">{line}</div>;
                            })}
                        </div>
                    </div>

                    {/* SPEAKING METRICS */}
                    <div className="metric-section">
                        <h4>🎤 Speaking Metrics</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Speaking Time:</span>
                                <span className="metric-value">
                                    {speakingTime !== 'N/A' ? `${typeof speakingTime === 'number' ? speakingTime.toFixed(1) : speakingTime}s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Total User Turn Time:</span>
                                <span className="metric-value">
                                    {totalUserTurnTime !== 'N/A' ? `${typeof totalUserTurnTime === 'number' ? totalUserTurnTime.toFixed(1) : totalUserTurnTime}s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Forced Silence (System Wait):</span>
                                <span className="metric-value">
                                    {forcedSilenceTime !== 'N/A' ? `${typeof forcedSilenceTime === 'number' ? forcedSilenceTime.toFixed(1) : forcedSilenceTime}s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item highlight">
                                <span className="metric-label">Available Speaking Time:</span>
                                <span className="metric-value">
                                    {availableSpeakingTime !== 'N/A' ? `${typeof availableSpeakingTime === 'number' ? availableSpeakingTime.toFixed(1) : availableSpeakingTime}s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Silence During Turn:</span>
                                <span className="metric-value">
                                    {silenceDuringTurn !== 'N/A' ? `${typeof silenceDuringTurn === 'number' ? silenceDuringTurn.toFixed(1) : silenceDuringTurn}s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Session Duration:</span>
                                <span className="metric-value">
                                    {sessionDuration !== 'N/A' ? `${typeof sessionDuration === 'number' ? sessionDuration.toFixed(1) : sessionDuration}s` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* SPEAKING RATIO & WPM */}
                    <div className="metric-section">
                        <h4>📊 Fluency Metrics</h4>
                        <div className="metrics-list">
                            <div className="metric-item highlight">
                                <span className="metric-label">Speaking Ratio:</span>
                                <span className="metric-value highlight-value">
                                    {speakingRatio !== 'N/A' ? `${(typeof speakingRatio === 'number' ? speakingRatio * 100 : speakingRatio).toFixed(1)}%` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Words Per Minute (WPM):</span>
                                <span className="metric-value">
                                    {wpm !== 'N/A' ? `${typeof wpm === 'number' ? wpm.toFixed(1) : wpm} wpm` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Articulation Rate:</span>
                                <span className="metric-value">
                                    {articulationRate !== 'N/A' ? `${typeof articulationRate === 'number' ? articulationRate.toFixed(2) : articulationRate} words/s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Avg Response Latency:</span>
                                <span className="metric-value">
                                    {avgResponseLatency !== 'N/A' ? `${typeof avgResponseLatency === 'number' ? avgResponseLatency.toFixed(2) : avgResponseLatency}s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Avg Pause Duration:</span>
                                <span className="metric-value">
                                    {avgPauseDuration !== 'N/A' ? `${typeof avgPauseDuration === 'number' ? avgPauseDuration.toFixed(2) : avgPauseDuration}s` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Pause Count:</span>
                                <span className="metric-value">{pauseCount !== 'N/A' ? pauseCount : 'N/A'}</span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Long Pauses ({'>'}5s):</span>
                                <span className="metric-value">{longPauseCount !== 'N/A' ? longPauseCount : 'N/A'}</span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Hesitation Rate:</span>
                                <span className="metric-value">
                                    {hesitationRate !== 'N/A' ? `${typeof hesitationRate === 'number' ? hesitationRate.toFixed(2) : hesitationRate}/min` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* CONTENT QUALITY */}
                    <div className="metric-section">
                        <h4>📋 Content Quality</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Semantic Similarity:</span>
                                <span className="metric-value true-value">
                                    {semanticSimilarity !== 'N/A' ? `${(typeof semanticSimilarity === 'number' ? semanticSimilarity * 100 : semanticSimilarity).toFixed(1)}%` : 'N/A'}
                                    <span className="value-note"> (raw cosine)</span>
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Keyword Coverage:</span>
                                <span className="metric-value true-value">
                                    {keywordCoverage !== 'N/A' ? `${(typeof keywordCoverage === 'number' ? keywordCoverage * 100 : keywordCoverage).toFixed(1)}%` : 'N/A'}
                                    <span className="value-note"> (stop words filtered)</span>
                                </span>
                            </div>
                            <div className="metric-item highlight">
                                <span className="metric-label">Overall Relevance:</span>
                                <span className="metric-value highlight-value">
                                    {overallRelevance !== 'N/A' ? `${(typeof overallRelevance === 'number' ? overallRelevance * 100 : overallRelevance).toFixed(1)}%` : 'N/A'}
                                    <span className="value-note"> (80/20 weighted)</span>
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Questions Answered:</span>
                                <span className="metric-value">{questionsAnswered !== 'N/A' ? questionsAnswered : 'N/A'}</span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Total Words:</span>
                                <span className="metric-value">{totalWords !== 'N/A' ? totalWords : 'N/A'}</span>
                            </div>
                        </div>
                    </div>

                    {/* VOICE ANALYSIS */}
                    {(pitchMean !== 'N/A' || pitchStability !== 'N/A') && (
                        <div className="metric-section">
                            <h4>🎤 Voice Analysis</h4>
                            <div className="metrics-list">
                                {pitchMean !== 'N/A' && (
                                    <div className="metric-item">
                                        <span className="metric-label">Average Pitch:</span>
                                        <span className="metric-value">
                                            {typeof pitchMean === 'number' ? pitchMean.toFixed(1) : pitchMean} Hz
                                        </span>
                                    </div>
                                )}
                                {pitchRange !== 'N/A' && (
                                    <div className="metric-item">
                                        <span className="metric-label">Pitch Range:</span>
                                        <span className="metric-value">
                                            {typeof pitchRange === 'number' ? pitchRange.toFixed(1) : pitchRange} Hz
                                        </span>
                                    </div>
                                )}
                                {pitchStability !== 'N/A' && (
                                    <div className="metric-item">
                                        <span className="metric-label">Stability:</span>
                                        <span className="metric-value">
                                            <span className={`stability-badge ${pitchStability > 80 ? 'excellent' :
                                                pitchStability > 60 ? 'good' :
                                                    pitchStability > 40 ? 'warning' : 'poor'
                                                }`}>
                                                {typeof pitchStability === 'number' ? pitchStability.toFixed(1) : pitchStability}%
                                            </span>
                                        </span>
                                    </div>
                                )}
                                {pitchStd !== 'N/A' && (
                                    <div className="metric-item">
                                        <span className="metric-label">Variation (σ):</span>
                                        <span className="metric-value">
                                            {typeof pitchStd === 'number' ? pitchStd.toFixed(1) : pitchStd} Hz
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Q&A Details */}
                    {qaPairs.length > 0 && (
                        <div className="metric-section full-width">
                            <h4>📋 Question & Answer Details</h4>
                            <div className="qa-details">
                                <h5 onClick={() => toggleQAExpanded('all')} className="qa-toggle">
                                    {expandedQA.all ? '▼' : '▶'} View Detailed Q&A Analysis
                                </h5>
                                {expandedQA.all && (
                                    <div className="qa-list">
                                        {qaPairs.map((pair, idx) => (
                                            <div key={idx} className="qa-pair">
                                                <div className="qa-question"><strong>Q{idx + 1}:</strong> {pair.question}</div>
                                                <div className="qa-answer"><strong>Your Answer:</strong> {pair.answer}</div>
                                                {pair.expected_answer && <div className="qa-expected"><strong>Expected:</strong> {pair.expected_answer}</div>}
                                                <div className="qa-scores">
                                                    <span className="qa-score semantic">Semantic: {(pair.similarity * 100).toFixed(1)}%</span>
                                                    <span className="qa-score keyword">Keyword: {(pair.keyword_coverage * 100).toFixed(1)}%</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Processing Details */}
                    <div className="metric-section">
                        <h4>⚙️ Processing</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Analysis Valid:</span>
                                <span className="metric-value">{data.analysis_valid ? '✅ Yes' : '❌ No'}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="metrics-actions">
                    <button className="restart-btn" onClick={handleNewInterview}>
                        🔄 Start New Interview
                    </button>
                </div>
            </div>
        );
    };

    // ── PRE-INTERVIEW hero ────────────────────────────────────────────────────
    if (!started && !showMetrics) {
        return (
            <>
                <Header user={user} onLogout={onLogout} title="Agentic Interview" />
                <div className="agentic-pre-screen">
                    <div className="pre-avatar-hero">
                        <div className="pre-avatar-canvas">
                            <AvatarViewer persona={persona} isSpeaking={false} />
                        </div>
                        <div className="pre-avatar-namecard">
                            <div className="pre-avatar-badge" style={{ '--company-color': persona.companyColor }}>
                                <span className="pre-avatar-company">{persona.company}</span>
                            </div>
                            <h2 className="pre-avatar-name">{persona.name}</h2>
                            <p className="pre-avatar-title">{persona.title}</p>
                            <p className="pre-avatar-style">
                                <i className="fas fa-quote-left"></i>
                                {' '}Interviewing style: <em>{persona.style}</em>
                            </p>
                        </div>
                    </div>

                    <div className="pre-interview-info">
                        <h1>Ready for your<br /><span className="gradient-text">Agentic Interview?</span></h1>
                        <p className="pre-desc">
                            A fully autonomous AI interview. Your interviewer listens, adapts in real time,
                            and provides a detailed research-grade performance analysis at the end.
                        </p>

                        <div className="pre-tips">
                            <div className="pre-tip"><i className="fas fa-microphone"></i> Speak clearly — we transcribe live</div>
                            <div className="pre-tip"><i className="fas fa-clock"></i> 30 minutes max per session</div>
                            <div className="pre-tip"><i className="fas fa-robot"></i> AI adapts to your skill level</div>
                        </div>

                        <div className="pre-connection">
                            <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
                            <span>{isConnected ? 'Ready to start' : 'Connecting…'}</span>
                        </div>

                        <button
                            className="agentic-start-btn"
                            onClick={startAgenticInterview}
                            disabled={!isConnected}
                        >
                            <i className="fas fa-microphone"></i>
                            Start Interview with {persona.name.split(' ')[0]}
                        </button>
                    </div>
                </div>
            </>
        );
    }

    // ── POST-INTERVIEW metrics ────────────────────────────────────────────────
    if (showMetrics) {
        return (
            <>
                <Header user={user} onLogout={onLogout} title="Agentic Interview" />
                <div className="agentic-container">{renderMetrics()}</div>
            </>
        );
    }

    // ── DURING INTERVIEW — split layout ───────────────────────────────────────
    return (
        <>
            <Header user={user} onLogout={onLogout} title="Agentic Interview" />

            <div className="agentic-split-layout">
                {/* LEFT: Avatar panel */}
                <div className="avatar-panel">
                    <div className="avatar-panel-canvas">
                        <AvatarViewer persona={persona} isSpeaking={isInterviewerSpeaking} />
                    </div>

                    <div className={`avatar-speaking-badge ${isInterviewerSpeaking ? 'active' : ''}`}>
                        {isInterviewerSpeaking ? (
                            <>
                                <div className="speaking-dots">
                                    <span /><span /><span />
                                </div>
                                {persona.name.split(' ')[0]} is speaking…
                            </>
                        ) : currentTurn === 'USER' ? (
                            <><i className="fas fa-ear" /> Listening to you…</>
                        ) : (
                            <><i className="fas fa-clock" /> Thinking…</>
                        )}
                    </div>

                    <div className="avatar-namecard">
                        <div className="avatar-namecard-avatar" style={{ background: `linear-gradient(135deg, ${persona.avatarColor}, ${persona.accentColor})` }}>
                            {persona.initials}
                        </div>
                        <div className="avatar-namecard-info">
                            <strong>{persona.name}</strong>
                            <span>{persona.title}</span>
                            <div className="avatar-company-tag" style={{ '--company-color': persona.companyColor }}>
                                {persona.company}
                            </div>
                        </div>
                    </div>
                </div>

                {/* RIGHT: Chat panel */}
                <div className="chat-panel">
                    <div className="status-bar">
                        <div className="connection-indicator">
                            <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
                            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
                        </div>
                        <div className="turn-indicator">
                            <div className={`turn-dot ${currentTurn === 'USER' ? 'your-turn' : 'their-turn'}`} />
                            <span>
                                {currentTurn === 'USER' ? '🎤 Your turn' :
                                    currentTurn === 'INTERVIEWER' ? '🗣️ Interviewer speaking' : 'Waiting…'}
                            </span>
                        </div>
                        <div className="timer">⏱️ {timeRemaining}</div>
                    </div>

                    {/* Pitch Analysis Graph */}
                    {livePitch && livePitch.mean > 0 && (
                        <div className="pitch-section">
                            <PitchGraph
                                pitchHistory={pitchHistory || []}
                                stabilityHistory={stabilityHistory || []}
                                pitchTimestamps={pitchTimestamps || []}
                                livePitch={livePitch}
                            />
                        </div>
                    )}

                    {/* Session strategy */}
                    {sessionPlan && (
                        <div className="strategic-plan">
                            <h4>🎯 Session Strategy</h4>
                            <div className="plan-badges">
                                <span className="plan-badge primary">Focus: {sessionPlan.primary_focus}</span>
                                <span className="plan-badge strategy">Strategy: {sessionPlan.strategy}</span>
                                <span className="plan-badge target">Target: +{(sessionPlan.target_mastery_improvement * 100).toFixed(0)}%</span>
                            </div>
                            <div className="plan-details">
                                {Object.entries(sessionPlan.estimated_questions || {}).map(([topic, count]) => (
                                    <div key={topic} className="plan-topic">
                                        <span>{topic}</span><span>{count} questions</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Chat transcript */}
                    <div className="agentic-transcript">
                        <h3>Interview Conversation</h3>
                        <div className="chat-box" ref={chatBoxRef}>
                            {messages.length === 0 && !liveTranscript && (
                                <div className="chat-placeholder">Waiting for interview to begin…</div>
                            )}
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`chat-bubble ${msg.role}`}>
                                    <div className="bubble-header">
                                        {msg.role === 'interviewer'
                                            ? `🎙️ ${persona.name.split(' ')[0]}`
                                            : '👤 You'}
                                    </div>
                                    <div className="bubble-text">{msg.text}</div>
                                </div>
                            ))}
                            {liveTranscript && currentTurn === 'USER' && (
                                <div className="chat-bubble user live">
                                    <div className="bubble-header">👤 You <span className="live-indicator">(speaking…)</span></div>
                                    <div className="bubble-text typing">{liveTranscript}<span className="typing-cursor" /></div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Controls */}
                    <div className="agentic-controls">
                        <div className="agentic-status">{status}</div>
                        <button className="agentic-exit-btn" onClick={handleEndInterview}>
                            <i className="fas fa-stop-circle" /> End Interview
                        </button>
                    </div>

                    {error && <div className="agentic-error">⚠️ {error}</div>}
                </div>
            </div>
        </>
    );
};

export default AgenticInterview;