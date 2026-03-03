import React, { useState, useEffect, useRef } from 'react';
import Header from '../Layout/Header';
import { useInterviewStreaming } from '../../hooks/useInterviewStreaming';
import AvatarViewer from './AvatarViewer';
import { getPersona } from './interviewerPersonas';
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
        sessionPlan
    } = useInterviewStreaming(user.id);

    const [started, setStarted] = useState(false);
    const [showMetrics, setShowMetrics] = useState(false);
    const [expandedQA, setExpandedQA] = useState({});
    const [stressTestMode, setStressTestMode] = useState(false);
    const [persona] = useState(() => getPersona(getSessionCount())); // lock persona for this mount
    const chatBoxRef = useRef(null);

    // Auto-scroll chat
    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    }, [messages, liveTranscript]);

    // Show metrics when interview is done
    useEffect(() => {
        if (interviewDone && analysis) {
            setShowMetrics(true);
        }
    }, [interviewDone, analysis]);

    const startAgenticInterview = async () => {
        setStarted(true);
        setShowMetrics(false);
        startRecording(stressTestMode);
    };

    const handleEndInterview = () => {
        stopRecording();
        setShowMetrics(true);
    };

    const handleNewInterview = () => {
        incrementSessionCount();
        window.location.reload(); // reload to get fresh persona + fresh streaming hook
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const toggleQAExpanded = (index) => {
        setExpandedQA(prev => ({ ...prev, [index]: !prev[index] }));
    };

    // Helper function to safely get metrics values from different possible sources
    const getMetricValue = (key, defaultValue = 'N/A') => {
        // Check metrics object first
        if (metrics && metrics[key] !== undefined && metrics[key] !== null) {
            return metrics[key];
        }
        // Check analysis.metrics
        if (analysis && analysis.metrics && analysis.metrics[key] !== undefined) {
            return analysis.metrics[key];
        }
        // Check analysis directly
        if (analysis && analysis[key] !== undefined) {
            return analysis[key];
        }
        return defaultValue;
    };

    // ── Metrics panel (SHOWING ONLY REQUESTED METRICS) ──────────────────────
    const renderMetrics = () => {
        if (!analysis && !metrics) return null;

        const data = analysis || {};
        const metricsData = metrics || data.metrics || {};

        // 🔥 Parse speech_metrics JSON
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

        // Helper to get values from all possible sources
        const getValue = (key, defaultValue = 'N/A') => {
            if (metricsData[key] !== undefined && metricsData[key] !== null) return metricsData[key];
            if (data.metrics && data.metrics[key] !== undefined) return data.metrics[key];
            if (data[key] !== undefined) return data[key];
            if (speechMetrics[key] !== undefined) return speechMetrics[key];
            return defaultValue;
        };

        // Get speaking ratio
        const speakingRatio = getValue('speaking_ratio') || getValue('speaking_time_ratio');

        // ✅ REQUESTED METRICS ONLY
        const speakingTime = getValue('speaking_time');
        const silenceTime = getValue('silence_time');
        const wpm = getValue('wpm');
        const articulationRate = getValue('articulation_rate');
        const avgResponseLatency = getValue('avg_response_latency');
        const avgSemantic = getValue('avg_semantic_similarity') || 0;
        const avgKeyword = getValue('avg_keyword_coverage') || 0;
        const totalDuration = getValue('session_duration') || getValue('total_duration');

        // Calculate overall relevance
        let overallRelevance = getValue('overall_relevance');
        if (overallRelevance === 'N/A' && avgSemantic !== 'N/A') {
            const semantic = typeof avgSemantic === 'number' ? avgSemantic : 0;
            const keyword = typeof avgKeyword === 'number' ? avgKeyword : 0;
            overallRelevance = (semantic * 0.8) + (keyword * 0.2);
        }

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

                <h3>📊 Interview Analysis</h3>

                <div className="metrics-explanation">
                    <p><strong>All scores are raw, unaltered values (0.00 to 1.00):</strong></p>
                    <ul>
                        <li><strong>Semantic Similarity</strong> = True cosine similarity between your answer and expected answer</li>
                        <li><strong>Keyword Coverage</strong> = Percentage of technical terms you used (stop words filtered)</li>
                        <li><strong>Overall Relevance</strong> = 80% semantic + 20% keyword (weighted average)</li>
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

                    {/* ✅ SPEAKING TIME */}
                    <div className="metric-section">
                        <h4>⏱️ Speaking Time</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Speaking Time:</span>
                                <span className="metric-value">
                                    {speakingTime !== 'N/A' ? `${typeof speakingTime === 'number' ? speakingTime.toFixed(1) : speakingTime}s` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* ✅ SILENCE TIME (Thinking Time) */}
                    <div className="metric-section">
                        <h4>⏸️ Silence (Thinking Time)</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Silence Time:</span>
                                <span className="metric-value">
                                    {silenceTime !== 'N/A' ? `${typeof silenceTime === 'number' ? silenceTime.toFixed(1) : silenceTime}s` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* ✅ SPEAKING RATIO */}
                    <div className="metric-section">
                        <h4>📊 Speaking Ratio</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Speaking Ratio:</span>
                                <span className="metric-value">
                                    {speakingRatio !== 'N/A' ? `${(typeof speakingRatio === 'number' ? speakingRatio * 100 : speakingRatio).toFixed(1)}%` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Session Duration:</span>
                                <span className="metric-value">
                                    {totalDuration !== 'N/A' ? `${typeof totalDuration === 'number' ? totalDuration.toFixed(1) : totalDuration}s` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* ✅ WPM */}
                    <div className="metric-section">
                        <h4>⚡ Words Per Minute</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">WPM:</span>
                                <span className="metric-value">
                                    {wpm !== 'N/A' ? `${typeof wpm === 'number' ? wpm.toFixed(1) : wpm} wpm` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* ✅ ARTICULATION RATE */}
                    <div className="metric-section">
                        <h4>🗣️ Articulation Rate</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Articulation Rate:</span>
                                <span className="metric-value">
                                    {articulationRate !== 'N/A' ? `${typeof articulationRate === 'number' ? articulationRate.toFixed(2) : articulationRate} words/s` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* ✅ RESPONSE LATENCY */}
                    <div className="metric-section">
                        <h4>⚡ Response Latency</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Avg Response Latency:</span>
                                <span className="metric-value">
                                    {avgResponseLatency !== 'N/A' ? `${typeof avgResponseLatency === 'number' ? avgResponseLatency.toFixed(2) : avgResponseLatency}s` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* ✅ SEMANTIC SCORE & KEYWORD COVERAGE */}
                    <div className="metric-section">
                        <h4>📋 Content Quality</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Semantic Similarity:</span>
                                <span className="metric-value true-value">
                                    {avgSemantic !== 'N/A' ? `${(typeof avgSemantic === 'number' ? avgSemantic * 100 : avgSemantic).toFixed(1)}%` : 'N/A'}
                                    <span className="value-note"> (raw cosine)</span>
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Keyword Coverage:</span>
                                <span className="metric-value true-value">
                                    {avgKeyword !== 'N/A' ? `${(typeof avgKeyword === 'number' ? avgKeyword * 100 : avgKeyword).toFixed(1)}%` : 'N/A'}
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
                        </div>
                    </div>

                    {/* Q&A Details (Optional - Keep if you want) */}
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

                    {/* Processing Details (Minimal) */}
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
                    {/* Avatar hero */}
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

                    {/* Info + Start */}
                    <div className="pre-interview-info">
                        <h1>Ready for your<br /><span className="gradient-text">Agentic Interview?</span></h1>
                        <p className="pre-desc">
                            A fully autonomous AI interview. Your interviewer listens, adapts in real time,
                            and provides a detailed performance analysis at the end.
                        </p>

                        <div className="pre-tips">
                            <div className="pre-tip"><i className="fas fa-microphone"></i> Speak clearly — we transcribe live</div>
                            <div className="pre-tip"><i className="fas fa-clock"></i> 30 minutes max per session</div>
                            <div className="pre-tip"><i className="fas fa-robot"></i> AI adapts to your skill level</div>
                        </div>

                        <div className="agentic-mode-toggles" style={{ margin: '20px 0', padding: '15px', background: 'var(--bg-elevated)', borderRadius: '12px', border: '1px solid var(--border)' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', fontSize: '1.05rem', color: 'var(--text-primary)' }}>
                                <input
                                    type="checkbox"
                                    checked={stressTestMode}
                                    onChange={(e) => setStressTestMode(e.target.checked)}
                                    style={{ width: '20px', height: '20px', accentColor: persona.accentColor }}
                                />
                                <strong>Stress Test Mode</strong>
                                <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginLeft: '10px' }}>(AI actively challenges your answers)</span>
                            </label>
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

                {/* ── LEFT: Avatar panel ─────────────────────────────── */}
                <div className="avatar-panel">
                    <div className="avatar-panel-canvas">
                        <AvatarViewer persona={persona} isSpeaking={isInterviewerSpeaking} />
                    </div>

                    {/* Speaking status indicator */}
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

                    {/* Persona namecard */}
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

                {/* ── RIGHT: Chat panel ──────────────────────────────── */}
                <div className="chat-panel">

                    {/* Status bar */}
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

                    {/* Debug */}
                    {process.env.NODE_ENV === 'development' && (
                        <div className="debug-info">
                            <div>Messages: {messages.length}</div>
                            <div>Turn: {currentTurn}</div>
                            <div>Speaking: {isInterviewerSpeaking ? 'YES' : 'no'}</div>
                        </div>
                    )}

                    {error && <div className="agentic-error">⚠️ {error}</div>}
                </div>
            </div>
        </>
    );
};

export default AgenticInterview;