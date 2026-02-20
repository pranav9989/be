import React, { useState, useEffect, useRef } from 'react';
import Header from '../Layout/Header';
import { useInterviewStreaming } from '../../hooks/useInterviewStreaming';
import './AgenticInterview.css';

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
    const [expandedQA, setExpandedQA] = useState({}); // Track which Q&A pairs are expanded
    const chatBoxRef = useRef(null);

    // Auto-scroll to bottom when new messages arrive
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
        startRecording();
    };

    const handleEndInterview = () => {
        stopRecording();
        setShowMetrics(true);
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    // Toggle expanded view for a Q&A pair
    const toggleQAExpanded = (index) => {
        setExpandedQA(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    };

    // Format metrics for display
    // Format metrics for display - NO SCALING, TRUE VALUES ONLY
    const renderMetrics = () => {
        if (!analysis && !metrics) return null;

        const data = analysis || {};
        const metricsData = metrics || data.metrics || {};

        // Get conversation from analysis or build from messages
        const conversation = data.conversation ||
            (messages && messages.length > 0 ?
                messages.map(m => `${m.role === 'interviewer' ? 'Interviewer' : 'You'}: ${m.text}`).join('\n\n') :
                finalTranscript || "No transcript available");

        // Get Q&A pairs if available
        const qaPairs = data.qa_pairs || [];

        return (
            <div className="metrics-panel">
                <h3>📊 Interview Analysis</h3>

                {/* 🔥 NEW: Explanation of metrics */}
                <div className="metrics-explanation">
                    <p><strong>All scores are raw, unaltered values (0.00 to 1.00):</strong></p>
                    <ul>
                        <li><strong>Semantic Similarity</strong> = True cosine similarity between your answer and expected answer</li>
                        <li><strong>Keyword Coverage</strong> = Percentage of technical terms you used (stop words filtered)</li>
                        <li><strong>Overall Relevance</strong> = 80% semantic + 20% keyword (weighted average)</li>
                    </ul>
                </div>

                <div className="metrics-grid">
                    {/* Transcript Section - Shows FULL conversation */}
                    <div className="metric-section full-width">
                        <h4>📝 Complete Interview Transcript</h4>
                        <div className="transcript-box full-conversation">
                            {conversation.split('\n\n').map((line, i) => {
                                if (line.startsWith('Interviewer:')) {
                                    return <div key={i} className="transcript-line interviewer-line">{line}</div>;
                                } else if (line.startsWith('User:')) {
                                    return <div key={i} className="transcript-line user-line">{line}</div>;
                                } else {
                                    return <div key={i} className="transcript-line">{line}</div>;
                                }
                            })}
                        </div>
                    </div>

                    {/* Performance Metrics */}
                    <div className="metric-section">
                        <h4>📈 Performance Metrics</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Words Per Minute:</span>
                                <span className="metric-value">
                                    {metricsData.wpm ? `${metricsData.wpm.toFixed(1)} wpm` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Speaking Time Ratio:</span>
                                <span className="metric-value">
                                    {metricsData.speaking_time_ratio ? `${(metricsData.speaking_time_ratio * 100).toFixed(1)}%` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Pause Ratio:</span>
                                <span className="metric-value">
                                    {metricsData.pause_ratio ? `${(metricsData.pause_ratio * 100).toFixed(1)}%` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Long Pauses (&gt;5s):</span>
                                <span className="metric-value">
                                    {metricsData.long_pause_count || 0}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Total Duration:</span>
                                <span className="metric-value">
                                    {data.total_duration ? `${data.total_duration.toFixed(1)}s` : 'N/A'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Q&A Performance Section */}
                    <div className="metric-section">
                        <h4>📋 Q&A Performance</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Questions Answered:</span>
                                <span className="metric-value">
                                    {metricsData.questions_answered || data.question_count || 0}
                                </span>
                            </div>

                            {/* 🔥 TRUE values - NO SCALING */}
                            <div className="metric-item">
                                <span className="metric-label">Avg Semantic Similarity:</span>
                                <span className="metric-value true-value">
                                    {(metricsData.avg_semantic_similarity !== undefined ?
                                        (metricsData.avg_semantic_similarity * 100).toFixed(1) :
                                        data.avg_semantic_similarity ? (data.avg_semantic_similarity * 100).toFixed(1) : 'N/A')}%
                                    <span className="value-note"> (raw cosine)</span>
                                </span>
                            </div>

                            <div className="metric-item">
                                <span className="metric-label">Avg Keyword Coverage:</span>
                                <span className="metric-value true-value">
                                    {(metricsData.avg_keyword_coverage !== undefined ?
                                        (metricsData.avg_keyword_coverage * 100).toFixed(1) :
                                        data.avg_keyword_coverage ? (data.avg_keyword_coverage * 100).toFixed(1) : 'N/A')}%
                                    <span className="value-note"> (stop words filtered)</span>
                                </span>
                            </div>

                            <div className="metric-item highlight">
                                <span className="metric-label">Overall Relevance:</span>
                                <span className="metric-value highlight-value">
                                    {(metricsData.overall_relevance !== undefined ?
                                        (metricsData.overall_relevance * 100).toFixed(1) :
                                        data.combined_relevance_score ? (data.combined_relevance_score * 100).toFixed(1) : 'N/A')}%
                                    <span className="value-note"> (80/20 weighted)</span>
                                </span>
                            </div>
                        </div>

                        {/* Detailed Q&A Pairs (Collapsible) - Show TRUE per-question values */}
                        {qaPairs.length > 0 && (
                            <div className="qa-details">
                                <h5 onClick={() => toggleQAExpanded('all')} className="qa-toggle">
                                    {expandedQA.all ? '▼' : '▶'} View Detailed Q&A Analysis
                                </h5>
                                {expandedQA.all && (
                                    <div className="qa-list">
                                        {qaPairs.map((pair, idx) => (
                                            <div key={idx} className="qa-pair">
                                                <div className="qa-question">
                                                    <strong>Q{idx + 1}:</strong> {pair.question}
                                                </div>
                                                <div className="qa-answer">
                                                    <strong>Your Answer:</strong> {pair.answer}
                                                </div>
                                                {pair.expected_answer && (
                                                    <div className="qa-expected">
                                                        <strong>Expected:</strong> {pair.expected_answer}
                                                    </div>
                                                )}

                                                {/* 🔥 TRUE per-question scores - NO SCALING */}
                                                <div className="qa-scores">
                                                    <span className="qa-score semantic">
                                                        Semantic: {(pair.similarity * 100).toFixed(1)}%
                                                        <span className="score-note"> raw cosine</span>
                                                    </span>
                                                    <span className="qa-score keyword">
                                                        Keyword: {(pair.keyword_coverage * 100).toFixed(1)}%
                                                        <span className="score-note"> filtered</span>
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Processing Info */}
                    <div className="metric-section">
                        <h4>⚙️ Processing Details</h4>
                        <div className="metrics-list">
                            <div className="metric-item">
                                <span className="metric-label">Processing Method:</span>
                                <span className="metric-value">
                                    {data.processing_method || 'incremental_fast'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Total Words:</span>
                                <span className="metric-value">
                                    {data.total_words || 0}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Analysis Valid:</span>
                                <span className="metric-value">
                                    {data.analysis_valid ? '✅ Yes' : '❌ No'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Restart Button */}
                <div className="metrics-actions">
                    <button
                        className="restart-btn"
                        onClick={() => {
                            setShowMetrics(false);
                            setStarted(false);
                        }}
                    >
                        🔄 Start New Interview
                    </button>
                </div>
            </div>
        );
    };

    return (
        <>
            <Header user={user} onLogout={onLogout} title="Agentic Interview" />

            <div className="agentic-container">
                <div className="agentic-header">
                    <h2>Agentic Interview</h2>
                    <p>
                        A fully autonomous interview. The AI interviewer asks questions,
                        listens to your answers, and adapts in real time.
                    </p>
                </div>

                {/* Show metrics or interview interface */}
                {showMetrics ? (
                    renderMetrics()
                ) : (
                    <>
                        {/* Connection & Status Bar */}
                        <div className="status-bar">
                            <div className="connection-indicator">
                                <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></div>
                                <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
                            </div>

                            {started && (
                                <>
                                    <div className="turn-indicator">
                                        <div className={`turn-dot ${currentTurn === 'USER' ? 'your-turn' : 'their-turn'}`}></div>
                                        <span>
                                            {currentTurn === 'USER' ? '🎤 Your turn' :
                                                currentTurn === 'INTERVIEWER' ? '🗣️ Interviewer speaking' :
                                                    'Waiting...'}
                                        </span>
                                    </div>

                                    <div className="timer">
                                        ⏱️ {timeRemaining}
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Strategic Plan Display */}
                        {sessionPlan && started && (
                            <div className="strategic-plan">
                                <h4>🎯 Session Strategy</h4>
                                <div className="plan-badges">
                                    <span className="plan-badge primary">
                                        Focus: {sessionPlan.primary_focus}
                                    </span>
                                    <span className="plan-badge strategy">
                                        Strategy: {sessionPlan.strategy}
                                    </span>
                                    <span className="plan-badge target">
                                        Target: +{(sessionPlan.target_mastery_improvement * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <div className="plan-details">
                                    {Object.entries(sessionPlan.estimated_questions || {}).map(([topic, count]) => (
                                        <div key={topic} className="plan-topic">
                                            <span>{topic}</span>
                                            <span>{count} questions</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Control Buttons */}
                        <div className="agentic-controls">
                            {!started ? (
                                <button
                                    className="agentic-start-btn"
                                    onClick={startAgenticInterview}
                                    disabled={!isConnected}
                                >
                                    🎤 Start Interview
                                </button>
                            ) : (
                                <button
                                    className="agentic-exit-btn"
                                    onClick={handleEndInterview}
                                >
                                    ⛔ End Interview
                                </button>
                            )}
                        </div>

                        {/* Status Message */}
                        <div className="agentic-status">
                            {status}
                        </div>

                        {/* Chat Conversation */}
                        {started && (
                            <div className="agentic-transcript">
                                <h3>Interview Conversation</h3>

                                <div className="chat-box" ref={chatBoxRef}>
                                    {messages.length === 0 && !liveTranscript && (
                                        <div className="chat-placeholder">
                                            Waiting for interview to begin...
                                        </div>
                                    )}

                                    {messages.map((msg, idx) => (
                                        <div
                                            key={idx}
                                            className={`chat-bubble ${msg.role}`}
                                        >
                                            <div className="bubble-header">
                                                {msg.role === "interviewer" ? "🎙️ Interviewer" : "👤 You"}
                                            </div>
                                            <div className="bubble-text">{msg.text}</div>
                                        </div>
                                    ))}

                                    {/* Live Transcript (while user is speaking) */}
                                    {liveTranscript && currentTurn === 'USER' && (
                                        <div className="chat-bubble user live">
                                            <div className="bubble-header">
                                                👤 You <span className="live-indicator">(speaking...)</span>
                                            </div>
                                            <div className="bubble-text typing">
                                                {liveTranscript}
                                                <span className="typing-cursor"></span>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Debug Info (only in development) */}
                                {process.env.NODE_ENV === 'development' && (
                                    <div className="debug-info">
                                        <div>Messages: {messages.length}</div>
                                        <div>Current Turn: {currentTurn}</div>
                                        <div>Live Transcript: {liveTranscript ? `${liveTranscript.length} chars` : 'none'}</div>
                                        <div>Recording: {isRecording ? 'Yes' : 'No'}</div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Error Display */}
                        {error && (
                            <div className="agentic-error">
                                ⚠️ {error}
                            </div>
                        )}
                    </>
                )}
            </div>
        </>
    );
};

export default AgenticInterview;