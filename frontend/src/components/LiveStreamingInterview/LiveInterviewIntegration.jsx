import React from 'react';
import { useInterviewStreaming } from '../../hooks/useInterviewStreaming';

const LiveInterviewIntegration = ({ userId }) => {
    const {
        isConnected,
        isRecording,
        liveTranscript,  // <-- LIVE AUDIO RECOGNITION ACCESS HERE
        liveWpm,         // <-- LIVE WPM ACCESS HERE
        finalAnalysis,   // <-- FINAL ANALYSIS ACCESS HERE
        startRecording,
        stopRecording,
        error
    } = useInterviewStreaming(userId);

    return (
        <div className="live-interview">
            {/* Connection Status */}
            <div className="status-bar">
                Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                {isRecording && ' | 🎤 Recording'}
            </div>

            {/* Control Buttons */}
            <div className="controls">
                <button
                    onClick={startRecording}
                    disabled={!isConnected || isRecording}
                >
                    Start Live Interview
                </button>

                <button
                    onClick={stopRecording}
                    disabled={!isRecording}
                >
                    Stop & Get Analysis
                </button>
            </div>

            {/* LIVE AUDIO RECOGNITION - THIS IS WHERE YOU ACCESS IT */}
            <div className="live-transcript-section">
                <h3>Live Speech Recognition:</h3>
                <div className="live-transcript-box">
                    {liveTranscript || 'Start speaking to see live transcription...'}
                </div>

                {liveWpm > 0 && (
                    <div className="live-wpm">
                        Current WPM: {liveWpm}
                    </div>
                )}
            </div>

            {/* Final Analysis Results */}
            {finalAnalysis && (
                <div className="final-analysis">
                    <h3>Complete Analysis:</h3>
                    <div className="scores">
                        Overall: {finalAnalysis.overall_score?.toFixed(1)}/100 |
                        Fluency: {finalAnalysis.fluency_score?.toFixed(1)}/100 |
                        Voice: {finalAnalysis.voice_quality_score?.toFixed(1)}/100
                    </div>
                </div>
            )}

            {/* Error Display */}
            {error && (
                <div className="error-message">
                    Error: {error}
                </div>
            )}
        </div>
    );
};

export default LiveInterviewIntegration;