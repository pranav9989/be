import React, { useState, useEffect } from 'react';
import Header from '../Layout/Header';
import { resumeAPI } from '../../services/api';
import './MockInterview.css';

const MockInterview = ({ user, onLogout, initialJobDescription = '', onBack }) => {
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [jobDescription, setJobDescription] = useState(initialJobDescription);
  const [questionCount, setQuestionCount] = useState(5);
  const [attemptNumber, setAttemptNumber] = useState(1);

  useEffect(() => {
    // Check if user has uploaded resume
    if (!user.resume_filename) {
      alert('Please upload your resume first to start a mock interview.');
      if (onBack) {
        onBack();
      } else {
        window.location.href = '/upload-resume';
      }
      return;
    }

    // Auto-generate questions if job description is provided
    if (initialJobDescription && !questions.length) {
      generateQuestions();
    }
  }, [user, initialJobDescription]);

  const generateQuestions = async () => {
    if (!jobDescription.trim()) {
      alert('Please provide a job description to generate targeted questions.');
      return;
    }

    setLoading(true);
    try {
      const response = await resumeAPI.generateResumeBasedQuestions({
        job_description: jobDescription,
        question_count: questionCount,
        variation_seed: `${attemptNumber}_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`
      });

      setQuestions(response.data.questions);
      setCurrentQuestionIndex(0);
      setAnswers({});
      setShowResults(false);
    } catch (error) {
      alert('Failed to generate questions. Please try again.');
      console.error('Error generating questions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerChange = (questionIndex, answer) => {
    setAnswers(prev => ({
      ...prev,
      [questionIndex]: answer
    }));
  };

  const nextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      // All questions answered, show results
      setShowResults(true);
      generateFeedback();
    }
  };

  const previousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const generateFeedback = async () => {
    // This could be enhanced to send answers to backend for AI feedback
    setFeedback('Great job completing the mock interview! Review your answers and consider how you can improve your responses.');
  };

  const resetInterview = async () => {
    setAttemptNumber(prev => prev + 1); // Increment attempt number for different questions
    await generateQuestions(); // Generate new questions
  };

  if (showResults) {
    return (
      <div className="mock-interview-container">
        <Header
          user={user}
          onLogout={onLogout}
          title="Mock Interview Results"
          showBack={true}
        />

        <main className="interview-main">
          <section className="results-section">
            <div className="results-header">
              <h2><i className="fas fa-trophy"></i> Interview Complete!</h2>
              <p>You've successfully completed your mock interview.</p>
            </div>

            <div className="feedback-card">
              <h3><i className="fas fa-comments"></i> Feedback</h3>
              <p>{feedback}</p>
            </div>

            <div className="answers-review">
              <h3><i className="fas fa-list-check"></i> Your Answers</h3>
              {questions.map((question, index) => (
                <div key={index} className="answer-item">
                  <h4>Q{index + 1}: {question.question}</h4>
                  <div className="answer-text">
                    {answers[index] || 'No answer provided'}
                  </div>
                </div>
              ))}
            </div>

            <div className="action-buttons">
              <button
                className="retry-btn"
                onClick={resetInterview}
              >
                <i className="fas fa-redo"></i> Try Again
              </button>
              <button
                className="new-interview-btn"
                onClick={() => window.location.href = '/upload-resume'}
              >
                <i className="fas fa-upload"></i> Upload New Resume
              </button>
            </div>
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="mock-interview-container">
        <Header
          user={user}
          onLogout={onLogout}
          title="Mock Interview"
          showBack={!!onBack}
          onBack={onBack}
        />

      <main className="interview-main">
        {!questions.length ? (
          <section className="setup-section">
            <h2><i className="fas fa-microphone"></i> {initialJobDescription ? 'Customize Your Mock Interview' : 'Start Your Mock Interview'}</h2>
            <p>{initialJobDescription ? 'Review and adjust your settings before starting.' : 'Get personalized interview questions based on your resume and job description.'}</p>

            <div className="setup-form">
              <div className="form-group">
                <label htmlFor="job-description">
                  <i className="fas fa-briefcase"></i> Job Description
                </label>
                <textarea
                  id="job-description"
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  placeholder="Paste the job description here to get questions tailored to this specific role..."
                  rows="6"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="question-count">
                  <i className="fas fa-question-circle"></i> Number of Questions
                </label>
                <select
                  id="question-count"
                  value={questionCount}
                  onChange={(e) => setQuestionCount(parseInt(e.target.value))}
                >
                  <option value={3}>3 Questions</option>
                  <option value={5}>5 Questions</option>
                  <option value={7}>7 Questions</option>
                  <option value={10}>10 Questions</option>
                </select>
              </div>

              <div className="setup-actions">
                <button
                  className="generate-btn"
                  onClick={generateQuestions}
                  disabled={loading || !jobDescription.trim()}
                >
                  {loading ? (
                    <>
                      <i className="fas fa-spinner fa-spin"></i> Generating Questions...
                    </>
                  ) : (
                    <>
                      <i className="fas fa-play"></i> Start Mock Interview
                    </>
                  )}
                </button>

                {initialJobDescription && onBack && (
                  <button
                    className="back-btn"
                    onClick={onBack}
                  >
                    <i className="fas fa-arrow-left"></i> Back to Analysis
                  </button>
                )}
              </div>
            </div>
          </section>
        ) : (
          <section className="interview-section">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}
              ></div>
            </div>

            <div className="question-counter">
              Question {currentQuestionIndex + 1} of {questions.length}
            </div>

            <div className="question-card">
              <div className="question-header">
                <span className="question-type">{questions[currentQuestionIndex]?.type}</span>
              </div>

              <div className="question-text">
                {questions[currentQuestionIndex]?.question}
              </div>

              <div className="answer-section">
                <textarea
                  value={answers[currentQuestionIndex] || ''}
                  onChange={(e) => handleAnswerChange(currentQuestionIndex, e.target.value)}
                  placeholder="Type your answer here..."
                  rows="6"
                />
              </div>

              <div className="navigation-buttons">
                <button
                  className="nav-btn previous"
                  onClick={previousQuestion}
                  disabled={currentQuestionIndex === 0}
                >
                  <i className="fas fa-arrow-left"></i> Previous
                </button>

                <button
                  className="nav-btn next"
                  onClick={nextQuestion}
                  disabled={!answers[currentQuestionIndex]?.trim()}
                >
                  {currentQuestionIndex === questions.length - 1 ? (
                    <>
                      <i className="fas fa-flag-checkered"></i> Finish Interview
                    </>
                  ) : (
                    <>
                      Next <i className="fas fa-arrow-right"></i>
                    </>
                  )}
                </button>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};

export default MockInterview;
