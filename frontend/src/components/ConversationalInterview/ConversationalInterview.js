import React, { useState, useRef, useEffect } from 'react';
import Header from '../Layout/Header';
import { audioAPI } from '../../services/api';
import './ConversationalInterview.css';


const ConversationalInterview = ({ user, onLogout }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interviewerResponse, setInterviewerResponse] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const [speechAnalysis, setSpeechAnalysis] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);


  const startRecording = async () => {
    try {
      setError("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Check what MIME types are supported
      const options = { mimeType: 'audio/webm' };
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        // Fallback to default if webm not supported
        options.mimeType = '';
      }

      mediaRecorderRef.current = new MediaRecorder(stream, options);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        // Stop all tracks to release microphone
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }

        if (audioChunksRef.current.length === 0) {
          setError("No audio data recorded. Please try again.");
          setIsProcessing(false);
          return;
        }

        setIsProcessing(true);
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        console.log('Audio blob created:', {
          size: audioBlob.size,
          type: audioBlob.type,
          chunks: audioChunksRef.current.length
        });

        if (audioBlob.size === 0) {
          setError("Recorded audio is empty. Please speak louder or check your microphone.");
          setIsProcessing(false);
          return;
        }

        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'interview_audio.webm');

        try {
          console.log('Sending audio to backend...');
          const response = await audioAPI.uploadAudio(formData);
          console.log('Backend response:', response.data);

          if (response.data.success) {
            setTranscript(response.data.transcribed_text || "");
            setInterviewerResponse(response.data.interviewer_response || "");
            setSpeechAnalysis(response.data.speech_analysis || null);
            console.log('Speech analysis data:', response.data.speech_analysis);
            console.log('Full response data:', response.data);
            setError(""); // Clear any previous errors
          } else {
            setError(response.data.message || response.data.error || "Failed to process audio");
          }
        } catch (error) {
          console.error('Error uploading audio:', error);
          const errorMessage = error.response?.data?.error ||
                              error.response?.data?.message ||
                              error.message ||
                              "Sorry, I couldn't process that. Please try again.";
          setError(errorMessage);
        } finally {
          setIsProcessing(false);
        }
      };

      // Start recording with timeslice to ensure data is collected
      mediaRecorderRef.current.start(1000); // Collect data every second
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setError("Could not access microphone. Please check permissions.");
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="conversational-interview-container">
      <Header
        user={user}
        onLogout={onLogout}
        title="Conversational AI Interviewer"
      />
      <main className="conversational-interview-main">
        <div className="interview-card">
          <h2>Start Your Conversational Interview</h2>
          <p>Practice speaking naturally with our AI interviewer.</p>
          <div className="controls">
            <button
              className={`btn ${isRecording ? 'btn-danger' : 'btn-primary'} ${isRecording ? 'recording-active' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
            >
              {isProcessing ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i> Processing...
                </>
              ) : isRecording ? (
                <>
                  <i className="fas fa-stop"></i> Stop Recording
                </>
              ) : (
                <>
                  <i className="fas fa-microphone"></i> Start Recording
                </>
              )}
            </button>
            {isRecording && (
              <div className="recording-indicator">
                <div className="pulse-ring"></div>
                <div className="pulse-ring-delayed"></div>
                <div className="recording-dot"></div>
                <span className="recording-text">Recording...</span>
              </div>
            )}
          </div>
          {error && (
            <div className="error-message">
              <i className="fas fa-exclamation-circle"></i> {error}
            </div>
          )}
          {transcript && (
            <div className="transcript-section">
              <h3>Your Transcript:</h3>
              <p>{transcript}</p>
            </div>
          )}
          {interviewerResponse && (
            <div className="interviewer-response-section">
              <h3>Interviewer:</h3>
              <p>{interviewerResponse}</p>
            </div>
          )}
          {/* Debug: Show if speechAnalysis exists */}
          {speechAnalysis && console.log('Rendering speech analysis:', speechAnalysis)}
          {speechAnalysis && (
            <div className="speech-analysis-section">
              <h3>Speech Analysis</h3>

              {/* Debug display */}
              <div style={{background: '#f0f0f0', padding: '10px', margin: '10px 0', border: '1px solid #ccc'}}>
                <strong>Debug Info:</strong><br/>
                Overall Score: {speechAnalysis.overall_score}<br/>
                Performance Level: {speechAnalysis.performance_level}<br/>
                Fluency Score: {speechAnalysis.fluency_score}<br/>
                Pitch Score: {speechAnalysis.pitch_score}<br/>
                Voice Quality Score: {speechAnalysis.voice_quality_score}<br/>
                WPM: {speechAnalysis.wpm}<br/>
              </div>

              {/* Overall Score */}
              <div className="analysis-score">
                <div className="score-circle" style={{
                  background: speechAnalysis.overall_score >= 80 ? '#10b981' :
                             speechAnalysis.overall_score >= 60 ? '#f59e0b' : '#ef4444'
                }}>
                  <span className="score-number">{Math.round(speechAnalysis.overall_score)}</span>
                  <span className="score-label">Overall</span>
                </div>
                <div className="performance-level">
                  <strong>{speechAnalysis.performance_level}</strong>
                </div>
              </div>

              {/* Detailed Scores */}
              <div className="detailed-scores">
                <div className="score-item">
                  <label>Fluency</label>
                  <div className="score-bar">
                    <div className="score-fill" style={{
                      width: `${speechAnalysis.fluency_score}%`,
                      backgroundColor: speechAnalysis.fluency_score >= 70 ? '#10b981' :
                                     speechAnalysis.fluency_score >= 50 ? '#f59e0b' : '#ef4444'
                    }}></div>
                  </div>
                  <span className="score-value">{Math.round(speechAnalysis.fluency_score)}</span>
                </div>

                <div className="score-item">
                  <label>Pitch</label>
                  <div className="score-bar">
                    <div className="score-fill" style={{
                      width: `${speechAnalysis.pitch_score}%`,
                      backgroundColor: speechAnalysis.pitch_score >= 70 ? '#10b981' :
                                     speechAnalysis.pitch_score >= 50 ? '#f59e0b' : '#ef4444'
                    }}></div>
                  </div>
                  <span className="score-value">{Math.round(speechAnalysis.pitch_score)}</span>
                </div>

                <div className="score-item">
                  <label>Voice Quality</label>
                  <div className="score-bar">
                    <div className="score-fill" style={{
                      width: `${speechAnalysis.voice_quality_score}%`,
                      backgroundColor: speechAnalysis.voice_quality_score >= 70 ? '#10b981' :
                                     speechAnalysis.voice_quality_score >= 50 ? '#f59e0b' : '#ef4444'
                    }}></div>
                  </div>
                  <span className="score-value">{Math.round(speechAnalysis.voice_quality_score)}</span>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="key-metrics">
                <div className="metric">
                  <span className="metric-label">Words/Minute:</span>
                  <span className="metric-value">{Math.round(speechAnalysis.wpm)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Pause Ratio:</span>
                  <span className="metric-value">{(speechAnalysis.pause_ratio * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Feedback */}
              <div className="feedback-section">
                {speechAnalysis.fluency_feedback && (
                  <p className="feedback-item"><strong>Fluency:</strong> {speechAnalysis.fluency_feedback}</p>
                )}
                {speechAnalysis.pitch_feedback && (
                  <p className="feedback-item"><strong>Pitch:</strong> {speechAnalysis.pitch_feedback}</p>
                )}
                {speechAnalysis.voice_quality_feedback && (
                  <p className="feedback-item"><strong>Voice Quality:</strong> {speechAnalysis.voice_quality_feedback}</p>
                )}
              </div>

              {/* Improvement Suggestions */}
              {speechAnalysis.improvement_suggestions && speechAnalysis.improvement_suggestions.length > 0 && (
                <div className="suggestions-section">
                  <h4>💡 Improvement Suggestions:</h4>
                  <ul>
                    {speechAnalysis.improvement_suggestions.map((suggestion, index) => (
                      <li key={index}>{suggestion}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default ConversationalInterview;
