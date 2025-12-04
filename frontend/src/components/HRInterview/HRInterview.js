import React, { useState, useEffect } from 'react';
import Header from '../Layout/Header';
import { hrAPI } from '../../services/api';
import './HRInterview.css';

const HRInterview = ({ user, onLogout }) => {
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [history, setHistory] = useState([]);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await hrAPI.getHistory('hr');
      setHistory(response.data.sessions || []);
    } catch (error) {
      console.error('Failed to load history:', error);
      showMessage('error', 'Failed to load session history');
    }
  };

  const generateQuestions = async () => {
    setLoading(true);
    try {
      const response = await hrAPI.generateQuestions();
      setQuestions(response.data.questions || []);
      setAnswers({});
      showMessage('success', 'Questions generated successfully!');
    } catch (error) {
      showMessage('error', 'Failed to generate questions');
    } finally {
      setLoading(false);
    }
  };

  const startInterview = async () => {
    if (questions.length === 0) {
      await generateQuestions();
    }
    setSessionStarted(true);
  };

  const handleAnswerChange = (index, answer) => {
    setAnswers(prev => ({
      ...prev,
      [index]: answer
    }));
  };

  const submitSession = async () => {
    if (Object.keys(answers).length === 0) {
      showMessage('error', 'Please answer at least one question');
      return;
    }

    setLoading(true);
    try {
      await hrAPI.saveSession({
        session_type: 'hr',
        questions: questions,
        answers: answers
      });
      showMessage('success', 'Session saved successfully!');
      setSessionStarted(false);
      setQuestions([]);
      setAnswers({});
      loadHistory();
    } catch (error) {
      showMessage('error', 'Failed to save session');
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };

  const QuestionCard = ({ question, index, answer, onAnswerChange }) => (
    <div className="question-card">
      <div className="question-header">
        <div className="question-number">{index + 1}</div>
        <div className="question-text">{question}</div>
      </div>
      <textarea
        className="answer-input"
        placeholder="Type your answer here... (Minimum 50 characters)"
        value={answer}
        onChange={(e) => onAnswerChange(index, e.target.value)}
        minLength="50"
      />
      <div className="character-count">
        {answer.length}/50 characters minimum
      </div>
    </div>
  );

  const SessionHistory = ({ history }) => (
    <div className="history-list">
      {history.map((session, index) => (
        <div key={index} className="history-item">
          <div className="session-date">
            {new Date(session.createdAt).toLocaleDateString()}
          </div>
          <div className="session-stats">
            Answered: {Object.keys(session.answers || {}).length}/{session.questions?.length || 0} questions
          </div>
          <button className="btn-view-session">
            View Details
          </button>
        </div>
      ))}
    </div>
  );

  return (
    <div className="hr-interview-container">
      <Header 
        user={user} 
        onLogout={onLogout} 
        title="HR Interview Practice" 
        showBack={true}
      />

      <main className="hr-main">
        {message.text && (
          <div className={`message ${message.type}`}>
            <i className={`fas ${message.type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}`}></i>
            {message.text}
          </div>
        )}

        {!sessionStarted ? (
          <section className="intro-section">
            <div className="intro-content">
              <h2>HR Interview Simulation</h2>
              <p>Practice behavioral and HR questions tailored to your profile and experience</p>
              
              <div className="action-buttons">
                <button 
                  onClick={startInterview} 
                  className="btn btn-primary"
                  disabled={loading}
                >
                  <i className="fas fa-play"></i> 
                  Start Interview
                </button>
                <button 
                  onClick={generateQuestions} 
                  className="btn btn-secondary"
                  disabled={loading}
                >
                  <i className="fas fa-sync"></i> 
                  Generate New Questions
                </button>
              </div>

              {loading && (
                <div className="loading">
                  <i className="fas fa-spinner fa-spin"></i>
                  <p>Generating questions...</p>
                </div>
              )}
            </div>
          </section>
        ) : (
          <section className="interview-section">
            <div className="interview-header">
              <h3>HR Interview Session</h3>
              <div className="progress">
                Answered: {Object.keys(answers).length}/{questions.length} questions
              </div>
            </div>

            <div className="questions-container">
              {questions.map((question, index) => (
                <QuestionCard
                  key={index}
                  question={question}
                  index={index}
                  answer={answers[index] || ''}
                  onAnswerChange={handleAnswerChange}
                />
              ))}
            </div>
            
            <div className="session-controls">
              <button 
                onClick={submitSession} 
                className="btn btn-primary" 
                disabled={loading || Object.keys(answers).length === 0}
              >
                <i className="fas fa-check"></i> 
                Submit Answers
              </button>
              <button 
                onClick={() => {
                  setSessionStarted(false);
                  setAnswers({});
                }} 
                className="btn btn-secondary"
                disabled={loading}
              >
                <i className="fas fa-times"></i> 
                Cancel Session
              </button>
            </div>
          </section>
        )}

        <section className="session-history">
          <h3>Previous Sessions</h3>
          {history.length === 0 ? (
            <div className="empty-history">
              <i className="fas fa-history"></i>
              <p>No previous sessions found.</p>
              <small>Complete your first interview session to see history here</small>
            </div>
          ) : (
            <SessionHistory history={history} />
          )}
        </section>
      </main>
    </div>
  );
};

export default HRInterview;