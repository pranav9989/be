import React from 'react';
import Header from '../Layout/Header';
import { useInterviewStreaming } from '../../hooks/useInterviewStreaming';
import './InterviewStreamer.css';

const InterviewStreamer = ({ userId, onLogout, user }) => {
    const {
        isConnected,
        isRecording,
        transcript,
        interviewDone,
        wpm,
        finalAnalysis,
        isFinalizing,
        status, // 🔥 ADDED: New status property
        startRecording,
        stopRecording,
        error
    } = useInterviewStreaming(userId);

    return (
        <div className="interview-streamer">
            <Header
                user={user}
                onLogout={onLogout}
                title="Live Streaming Interview"
            />

            <main className="streaming-main">
                {/* ---------------- STATUS ---------------- */}
                <div className="status">
                    <div className="status-item">
                        <span className="status-label">Connection:</span>
                        <span className="status-value">
                            {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                        </span>
                    </div>

                    <div className="status-item">
                        <span className="status-label">Status:</span>
                        <span className="status-value">
                            {status || 'Ready'} {/* 🔥 Shows status messages */}
                        </span>
                    </div>

                    <div className="status-item">
                        <span className="status-label">Recording:</span>
                        <span className="status-value">
                            {isRecording && !isFinalizing && '🎤 ON'}
                            {isFinalizing && '⏳ Finalizing...'}
                            {interviewDone && !isFinalizing && '✅ COMPLETED'}
                            {!isRecording && !isFinalizing && !interviewDone && '⏸️ OFF'}
                        </span>
                    </div>
                </div>

                {/* ---------------- CONTROLS ---------------- */}
                <div className="controls">
                    {!interviewDone ? (
                        <>
                            <button
                                onClick={startRecording}
                                disabled={!isConnected || isRecording || status.includes('Warming')}
                                className="start-btn"
                                title={status.includes('Warming') ? "Please wait while system warms up" : ""}
                            >
                                {status.includes('Warming') ? '🔥 Warming up...' : '🎤 Start Interview'}
                            </button>

                            <button
                                onClick={stopRecording}
                                disabled={!isRecording}
                                className="stop-btn"
                            >
                                ⏹️ Stop & Analyze
                            </button>
                        </>
                    ) : (
                        <button
                            onClick={startRecording}
                            disabled={!isConnected}
                            className="start-btn"
                        >
                            🔄 Start New Interview
                        </button>
                    )}
                </div>

                {/* ---------------- TRANSCRIPT ---------------- */}
                <div className="live-transcript-section">
                    <h3>
                        {interviewDone
                            ? 'Final Transcript'
                            : 'Live Speech Recognition'}
                    </h3>

                    <div className="live-transcript-box">
                        {transcript ||
                            (status.includes('Warming') ? '🔥 Warming up speech recognition...' :
                                status.includes('Starting') ? '🎤 Starting interview...' :
                                    interviewDone ? 'Analysis complete — see results below' :
                                        'Start speaking to see live transcription...')}
                    </div>

                    {!interviewDone && wpm > 0 && (
                        <div className="live-wpm">
                            Current WPM: {wpm}
                        </div>
                    )}
                </div>

                {/* ---------------- ANALYSIS RESULTS ---------------- */}
                {finalAnalysis && (
                    <div className="analysis-results">
                        <h3>Interview Analysis</h3>

                        {/* ===== OVERALL SCORE ===== */}
                        <div className="overall-score">
                            <h2>{finalAnalysis.overall_score?.toFixed(1) || 'N/A'}/100</h2>
                            <p className="performance-level">
                                {finalAnalysis.performance_level || 'Not Available'}
                            </p>
                        </div>

                        {/* ===== SCORE BREAKDOWN ===== */}
                        <div className="scores-grid">
                            <div className="score-card">
                                <div className="score-title">Fluency</div>
                                <div className="score-value">
                                    {finalAnalysis.fluency_score?.toFixed(1) || 'N/A'}
                                </div>
                            </div>
                            <div className="score-card">
                                <div className="score-title">Clarity</div>
                                <div className="score-value">
                                    {finalAnalysis.clarity_score?.toFixed(1) || 'N/A'}
                                </div>
                            </div>
                            <div className="score-card">
                                <div className="score-title">Pitch</div>
                                <div className="score-value">
                                    {finalAnalysis.pitch_score?.toFixed(1) || 'N/A'}
                                </div>
                            </div>
                            <div className="score-card">
                                <div className="score-title">Voice Quality</div>
                                <div className="score-value">
                                    {finalAnalysis.voice_quality_score?.toFixed(1) || 'N/A'}
                                </div>
                            </div>
                        </div>

                        {/* ===== CONTENT UNDERSTANDING ===== */}
                        <section className="analysis-section">
                            <h4>Content Understanding</h4>

                            <div className="metric">
                                <span>Answer Relevance</span>
                                <span>
                                    {finalAnalysis.semantic_similarity ?
                                        (finalAnalysis.semantic_similarity * 100).toFixed(1) + '%' : 'N/A'}
                                </span>
                            </div>

                            <div className="metric">
                                <span>Keyword Coverage</span>
                                <span>
                                    {finalAnalysis.keyword_coverage ?
                                        (finalAnalysis.keyword_coverage * 100).toFixed(1) + '%' : 'N/A'}
                                </span>
                            </div>
                        </section>

                        {/* ===== FLUENCY METRICS ===== */}
                        <section className="analysis-section">
                            <h4>Fluency Metrics</h4>

                            <div className="metric">
                                <span>Words per Minute</span>
                                <span>{finalAnalysis.wpm?.toFixed(1) || 'N/A'}</span>
                            </div>

                            <div className="metric">
                                <span>Pause Ratio</span>
                                <span>{finalAnalysis.pause_ratio?.toFixed(3) || 'N/A'}</span>
                            </div>

                            <div className="metric">
                                <span>Silence Ratio</span>
                                <span>
                                    {finalAnalysis.silence_ratio ?
                                        (finalAnalysis.silence_ratio * 100).toFixed(1) + '%' : 'N/A'}
                                </span>
                            </div>
                        </section>

                        {/* ===== DURATION METRICS ===== */}
                        <section className="analysis-section">
                            <h4>Duration</h4>

                            <div className="metric">
                                <span>Speaking Time</span>
                                <span>
                                    {finalAnalysis.speaking_time ?
                                        `${finalAnalysis.speaking_time.toFixed(1)}s` : 'N/A'}
                                </span>
                            </div>

                            <div className="metric">
                                <span>Total Duration</span>
                                <span>
                                    {finalAnalysis.total_duration ?
                                        `${finalAnalysis.total_duration.toFixed(1)}s` : 'N/A'}
                                </span>
                            </div>

                            <div className="metric">
                                <span>Engagement</span>
                                <span>
                                    {finalAnalysis.speaking_time && finalAnalysis.total_duration ?
                                        ((finalAnalysis.speaking_time / finalAnalysis.total_duration) * 100).toFixed(1) + '%' : 'N/A'}
                                </span>
                            </div>
                        </section>

                        {/* ===== VOICE ANALYSIS ===== */}
                        {(finalAnalysis.pitch_mean || finalAnalysis.pitch_range) && (
                            <section className="analysis-section">
                                <h4>Voice Analysis</h4>

                                {finalAnalysis.pitch_mean && (
                                    <div className="metric">
                                        <span>Average Pitch</span>
                                        <span>{finalAnalysis.pitch_mean.toFixed(1)} Hz</span>
                                    </div>
                                )}

                                {finalAnalysis.pitch_range && (
                                    <div className="metric">
                                        <span>Pitch Range</span>
                                        <span>{finalAnalysis.pitch_range.toFixed(1)} Hz</span>
                                    </div>
                                )}

                                {finalAnalysis.pitch_mean && finalAnalysis.pitch_std && (
                                    <div className="metric">
                                        <span>Pitch Stability</span>
                                        <span>
                                            {(finalAnalysis.pitch_std / finalAnalysis.pitch_mean).toFixed(2)}
                                        </span>
                                    </div>
                                )}
                            </section>
                        )}

                        {/* ===== AUDIO INFO ===== */}
                        <section className="analysis-section">
                            <h4>Audio Details</h4>

                            <div className="metric">
                                <span>Audio Saved</span>
                                <span className={finalAnalysis.audio_saved ? 'success' : 'warning'}>
                                    {finalAnalysis.audio_saved ? '✅ Yes' : '⚠️ No'}
                                </span>
                            </div>

                            <div className="metric">
                                <span>Chunks Processed</span>
                                <span>{finalAnalysis.chunks_processed || 0}</span>
                            </div>

                            <div className="metric">
                                <span>Buffered Chunks</span>
                                <span>{finalAnalysis.buffered_chunks || 0}</span>
                            </div>
                        </section>

                        {/* ===== IMPROVEMENT SUGGESTIONS ===== */}
                        {finalAnalysis.improvement_suggestions?.length > 0 && (
                            <section className="analysis-section suggestions-section">
                                <h4>Improvement Suggestions</h4>
                                <ul className="suggestions-list">
                                    {finalAnalysis.improvement_suggestions.map((item, idx) => (
                                        <li key={idx} className="suggestion-item">
                                            <span className="suggestion-icon">💡</span>
                                            <span className="suggestion-text">{item}</span>
                                        </li>
                                    ))}
                                </ul>
                            </section>
                        )}

                        {/* ===== DEBUG INFO (Optional) ===== */}
                        {process.env.NODE_ENV === 'development' && (
                            <details className="debug-info">
                                <summary>Debug Info</summary>
                                <pre>{JSON.stringify(finalAnalysis, null, 2)}</pre>
                            </details>
                        )}
                    </div>
                )}

                {/* ---------------- ERROR ---------------- */}
                {error && (
                    <div className="error-message">
                        <span className="error-icon">❌</span>
                        <span className="error-text">Error: {error}</span>
                    </div>
                )}
            </main>
        </div>
    );
};

export default InterviewStreamer;