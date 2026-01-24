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
                    <span>Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}</span>
                    <span>
                        Recording:&nbsp;
                        {isRecording && !isFinalizing && '🎤 ON'}
                        {isFinalizing && '⏳ Finalizing...'}
                        {interviewDone && !isFinalizing && '✅ COMPLETED'}
                        {!isRecording && !isFinalizing && !interviewDone && '⏸️ OFF'}
                    </span>
                </div>

                {/* ---------------- CONTROLS ---------------- */}
                <div className="controls">
                    {!interviewDone ? (
                        <>
                            <button
                                onClick={startRecording}
                                disabled={!isConnected || isRecording}
                                className="start-btn"
                            >
                                🎤 Start Interview
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
                            (interviewDone
                                ? 'Analysis complete — see results below'
                                : 'Start speaking to see live transcription...')}
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
                            <h2>{finalAnalysis.overall_score}/100</h2>
                            <p className="performance-level">
                                {finalAnalysis.performance_level}
                            </p>
                            <p className="performance-feedback">
                                {finalAnalysis.performance_feedback}
                            </p>
                        </div>

                        {/* ===== CONTENT UNDERSTANDING ===== */}
                        <section className="analysis-section">
                            <h4>Content Understanding</h4>

                            <div className="metric">
                                <span>Answer Relevance</span>
                                <span>
                                    {(finalAnalysis.semantic_similarity * 100).toFixed(1)}%
                                </span>
                            </div>

                            <div className="metric">
                                <span>Technical Coverage</span>
                                <span>
                                    {(finalAnalysis.keyword_coverage * 100).toFixed(1)}%
                                </span>
                            </div>
                        </section>

                        {/* ===== ENGAGEMENT ===== */}
                        <section className="analysis-section">
                            <h4>Engagement</h4>

                            <div className="metric">
                                <span>Speaking Time</span>
                                <span>{finalAnalysis.speaking_time.toFixed(1)} s</span>
                            </div>

                            <div className="metric">
                                <span>Total Duration</span>
                                <span>{finalAnalysis.total_duration.toFixed(1)} s</span>
                            </div>

                            <div className="metric">
                                <span>Engagement Ratio</span>
                                <span>
                                    {(
                                        (finalAnalysis.speaking_time /
                                            finalAnalysis.total_duration) *
                                        100
                                    ).toFixed(1)}%
                                </span>
                            </div>
                        </section>

                        {/* ===== VOICE STABILITY ===== */}
                        <section className="analysis-section">
                            <h4>Voice Stability</h4>

                            <div className="metric">
                                <span>Average Pitch</span>
                                <span>{finalAnalysis.pitch_mean.toFixed(1)} Hz</span>
                            </div>

                            <div className="metric">
                                <span>Pitch Range</span>
                                <span>{finalAnalysis.pitch_range.toFixed(1)} Hz</span>
                            </div>

                            <div className="metric">
                                <span>Pitch Variability</span>
                                <span>
                                    {(finalAnalysis.pitch_range /
                                        finalAnalysis.pitch_mean).toFixed(2)}
                                </span>
                            </div>
                        </section>

                        {/* ===== FEEDBACK ===== */}
                        {finalAnalysis.detailed_feedback?.length > 0 && (
                            <section className="analysis-section">
                                <h4>Detailed Feedback</h4>
                                <ul>
                                    {finalAnalysis.detailed_feedback.map((item, idx) => (
                                        <li key={idx}>{item}</li>
                                    ))}
                                </ul>
                            </section>
                        )}

                        {/* ===== IMPROVEMENTS ===== */}
                        {finalAnalysis.improvement_suggestions?.length > 0 && (
                            <section className="analysis-section">
                                <h4>Improvement Suggestions</h4>
                                <ul>
                                    {finalAnalysis.improvement_suggestions.map((item, idx) => (
                                        <li key={idx}>{item}</li>
                                    ))}
                                </ul>
                            </section>
                        )}
                    </div>
                )}

                {/* ---------------- ERROR ---------------- */}
                {error && (
                    <div className="error-message">
                        Error: {error}
                    </div>
                )}
            </main>
        </div>
    );
};

export default InterviewStreamer;
