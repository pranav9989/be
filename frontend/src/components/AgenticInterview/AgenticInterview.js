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
        isInterviewerSpeaking
    } = useInterviewStreaming(user.id);

    const [started, setStarted] = useState(false);
    const chatBoxRef = useRef(null);

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    }, [messages, liveTranscript]);

    const startAgenticInterview = async () => {
        setStarted(true);
        startRecording();
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
                            onClick={() => {
                                stopRecording();
                                setStarted(false);
                            }}
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
            </div>
        </>
    );
};

export default AgenticInterview;