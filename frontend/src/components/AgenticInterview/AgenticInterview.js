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
        interviewDone,        // Add this
        analysis,             // Add this
        finalTranscript,      // Add this
        metrics              // Add this
    } = useInterviewStreaming(user.id);

    const [started, setStarted] = useState(false);
    const [showMetrics, setShowMetrics] = useState(false); // Add this
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
        setShowMetrics(false); // Reset metrics display
        startRecording();
    };

    const handleEndInterview = () => {
        stopRecording();
        setShowMetrics(true); // Show metrics immediately
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    // Format metrics for display
    // Format metrics for display
    const renderMetrics = () => {
        if (!analysis && !metrics) return null;

        const data = analysis || {};
        const metricsData = metrics || data.metrics || {};

        return (
            <div className="metrics-panel">
                <h3>📊 Interview Analysis</h3>

                <div className="metrics-grid">
                    {/* Transcript Section */}
                    <div className="metric-section">
                        <h4>📝 Transcript</h4>
                        <div className="transcript-box">
                            {finalTranscript || data.transcript || "No transcript available"}
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

                            {/* 🔥 NEW: Q&A Metrics Section */}
                            <div className="metric-item" style={{ borderTop: '2px solid #3498db', marginTop: '10px', paddingTop: '10px' }}>
                                <span className="metric-label" style={{ fontWeight: 'bold', color: '#3498db' }}>📋 Q&A Performance:</span>
                                <span className="metric-value"></span>
                            </div>

                            <div className="metric-item">
                                <span className="metric-label">Questions Answered:</span>
                                <span className="metric-value">
                                    {metricsData.questions_answered || 0}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Avg Semantic Similarity:</span>
                                <span className="metric-value">
                                    {metricsData.avg_semantic_similarity ? `${(metricsData.avg_semantic_similarity * 100).toFixed(1)}%` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-label">Avg Keyword Coverage:</span>
                                <span className="metric-value">
                                    {metricsData.avg_keyword_coverage ? `${(metricsData.avg_keyword_coverage * 100).toFixed(1)}%` : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-item" style={{ backgroundColor: '#e8f4f8', borderRadius: '4px', padding: '8px' }}>
                                <span className="metric-label" style={{ fontWeight: 'bold' }}>Overall Relevance:</span>
                                <span className="metric-value" style={{ fontWeight: 'bold', color: '#2980b9' }}>
                                    {metricsData.overall_relevance ? `${(metricsData.overall_relevance * 100).toFixed(1)}%` : 'N/A'}
                                </span>
                            </div>

                            {/* Original metrics continue */}
                            <div className="metric-item">
                                <span className="metric-label">Total Duration:</span>
                                <span className="metric-value">
                                    {data.total_duration ? `${data.total_duration.toFixed(1)}s` : 'N/A'}
                                </span>
                            </div>
                        </div>
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