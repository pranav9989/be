import React, { useState } from 'react';
import Header from '../Layout/Header';
import { resumeAPI } from '../../services/api';
import './ResumeUpload.css';

const ResumeUpload = ({ user, onLogout }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [showMockInterview, setShowMockInterview] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  // Mock Interview States
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [interviewLoading, setInterviewLoading] = useState(false);
  const [showInterviewResults, setShowInterviewResults] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [questionCount, setQuestionCount] = useState(5);

  const handleFileSelect = (file) => {
    if (!isValidFile(file)) {
      showMessage('error', 'Please select a valid PDF, DOC, or DOCX file (max 16MB)');
      return;
    }
    setSelectedFile(file);
    setMessage({ type: '', text: '' });
  };

  const isValidFile = (file) => {
    const allowedTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    const maxSize = 16 * 1024 * 1024; // 16MB
    return allowedTypes.includes(file.type) && file.size <= maxSize;
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      showMessage('error', 'Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('resume', selectedFile);
    if (jobDescription.trim()) {
      formData.append('job_description', jobDescription.trim());
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      const response = await resumeAPI.upload(formData);
      setResults(response.data.data);
      setAnalysisComplete(true);
      showMessage('success', 'Resume uploaded and analyzed successfully!');
      setUploadProgress(100);
    } catch (error) {
      showMessage('error', error.response?.data?.message || 'Upload failed');
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 2000);
    }
  };

  // Mock Interview Functions
  const generateQuestions = async () => {
    if (!jobDescription.trim()) {
      showMessage('error', 'Please provide a job description to generate targeted questions.');
      return;
    }

    setInterviewLoading(true);
    try {
      const response = await resumeAPI.generateResumeBasedQuestions({
        job_description: jobDescription,
        question_count: questionCount,
        variation_seed: `${Date.now()}_${Math.random().toString(36).substring(2, 8)}`
      });

      setQuestions(response.data.questions);
      setCurrentQuestionIndex(0);
      setAnswers({});
      setShowInterviewResults(false);
      setShowMockInterview(true);
    } catch (error) {
      showMessage('error', 'Failed to generate questions. Please try again.');
      console.error('Error generating questions:', error);
    } finally {
      setInterviewLoading(false);
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
      setShowInterviewResults(true);
      setFeedback('Great job completing the mock interview! Review your answers and consider how you can improve your responses.');
    }
  };

  const previousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const resetInterview = async () => {
    setQuestions([]);
    setAnswers({});
    setShowInterviewResults(false);
    setCurrentQuestionIndex(0);
    setFeedback('');
  };

  const showMessage = (type, text) => {
    setMessage({ type, text });
  };

  // Mock Interview Component
  const MockInterviewSection = () => {
    if (showInterviewResults) {
      return (
        <div className="interview-results-section">
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
              className="back-to-analysis-btn"
              onClick={() => setShowMockInterview(false)}
            >
              <i className="fas fa-arrow-left"></i> Back to Analysis
            </button>
          </div>
        </div>
      );
    }

    if (!questions.length) {
      return (
        <div className="interview-setup-section">
          <h2><i className="fas fa-microphone"></i> Start Your Mock Interview</h2>
          <p>Get personalized interview questions based on your resume and job description.</p>

          <div className="setup-form">
            <div className="form-group">
              <label htmlFor="mock-job-description">
                <i className="fas fa-briefcase"></i> Job Description
              </label>
              <textarea
                id="mock-job-description"
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
                disabled={interviewLoading || !jobDescription.trim()}
              >
                {interviewLoading ? (
                  <>
                    <i className="fas fa-spinner fa-spin"></i> Generating Questions...
                  </>
                ) : (
                  <>
                    <i className="fas fa-play"></i> Start Mock Interview
                  </>
                )}
              </button>

              <button
                className="back-btn"
                onClick={() => setShowMockInterview(false)}
              >
                <i className="fas fa-arrow-left"></i> Back to Analysis
              </button>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="interview-section">
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
            <span className="question-type">{questions[currentQuestionIndex]?.type || 'Behavioral'}</span>
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
      </div>
    );
  };

  return (
    <div className="resume-upload-container">
      <Header
        user={user}
        onLogout={onLogout}
        title={showMockInterview ? "Mock Interview" : "Resume Upload"}
        showBack={true}
        onBack={showMockInterview ? () => setShowMockInterview(false) : null}
      />

      <main className="resume-main">
        {message.text && (
          <div className={`message ${message.type}`}>
            {message.text}
          </div>
        )}

        {!showMockInterview ? (
          <>
            <section className="upload-section">
              <h2>Upload Your Resume</h2>
              <p>Upload your resume to get personalized interview questions and skills analysis</p>

              <div
                className="upload-area"
                onClick={() => document.getElementById('file-input').click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.add('dragover');
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.remove('dragover');
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.classList.remove('dragover');
                  const files = e.dataTransfer.files;
                  if (files.length > 0) {
                    handleFileSelect(files[0]);
                  }
                }}
              >
                {selectedFile ? (
                  <>
                    <i className="fas fa-file-alt upload-icon"></i>
                    <div className="upload-text">Selected: {selectedFile.name}</div>
                    <div className="upload-subtext">
                      Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </>
                ) : (
                  <>
                    <i className="fas fa-cloud-upload-alt upload-icon"></i>
                    <div className="upload-text">Click to upload or drag and drop</div>
                    <div className="upload-subtext">PDF, DOC, or DOCX files up to 16MB</div>
                  </>
                )}
              </div>

              <input
                type="file"
                id="file-input"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
                accept=".pdf,.doc,.docx"
              />

              <div className="job-description-section">
                <label htmlFor="job-description" className="job-description-label">
                  <i className="fas fa-briefcase"></i> Job Description (Optional)
                </label>
                <textarea
                  id="job-description"
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  placeholder="Paste the job description here to get more targeted interview questions based on the specific role..."
                  className="job-description-textarea"
                  rows="4"
                />
                <small className="job-description-help">
                  Adding a job description helps generate more relevant interview questions tailored to the specific role.
                </small>
              </div>

              {uploadProgress > 0 && (
                <div className="progress-bar show">
                  <div
                    className="progress-fill"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
              )}

              {selectedFile && (
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="upload-btn"
                >
                  {uploading ? (
                    <>
                      <i className="fas fa-spinner fa-spin"></i> Uploading...
                    </>
                  ) : (
                    <>
                      <i className="fas fa-upload"></i> Upload Resume
                    </>
                  )}
                </button>
              )}
            </section>

            {analysisComplete && results && (
              <section className="results-section">
                <div className="results-header">
                  <h3><i className="fas fa-chart-line"></i> Resume Analysis Complete!</h3>
                  <p>Here's your personalized career insights:</p>
                </div>

                <div className="analysis-grid">
                  {/* Resume Analysis */}
                  <div className="analysis-card resume-analysis">
                    <h4><i className="fas fa-file-alt"></i> Resume Analysis</h4>

                    <div className="analysis-item">
                      <h5><i className="fas fa-cogs"></i> Skills Found</h5>
                      <div className="skills-list">
                        {results.skills && results.skills.length > 0 ? (
                          results.skills.map((skill, index) => (
                            <span key={index} className="skill-tag">{skill}</span>
                          ))
                        ) : (
                          <p className="no-data">No technical skills detected</p>
                        )}
                      </div>
                    </div>

                    <div className="analysis-item">
                      <h5><i className="fas fa-calendar-alt"></i> Experience</h5>
                      <p className="experience-text">
                        {results.experience_years || 0} year{(results.experience_years || 0) !== 1 ? 's' : ''} of experience
                      </p>
                    </div>

                    {results.projects && results.projects.length > 0 && (
                      <div className="analysis-item">
                        <h5><i className="fas fa-project-diagram"></i> Key Projects</h5>
                        <ul className="projects-list">
                          {results.projects.map((project, index) => (
                            <li key={index}>{project}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  {/* Job Fit Analysis - Only show if JD was provided */}
                  {results.job_fit_analysis && (
                    <div className="analysis-card job-fit-analysis">
                      <h4><i className="fas fa-bullseye"></i> Job Fit Analysis</h4>

                      <div className="match-score">
                        <div className="score-circle">
                          <span className="score-number">{results.job_fit_analysis.match_percentage}%</span>
                          <span className="score-label">Match</span>
                        </div>
                      </div>

                      <div className="analysis-item">
                        <h5><i className="fas fa-check-circle"></i> Matching Skills</h5>
                        <div className="skills-list">
                          {results.job_fit_analysis.matching_skills.length > 0 ? (
                            results.job_fit_analysis.matching_skills.map((skill, index) => (
                              <span key={index} className="skill-tag match">{skill}</span>
                            ))
                          ) : (
                            <p className="no-data">No matching skills found</p>
                          )}
                        </div>
                      </div>

                      {results.job_fit_analysis.missing_skills.length > 0 && (
                        <div className="analysis-item">
                          <h5><i className="fas fa-exclamation-triangle"></i> Skills to Develop</h5>
                          <div className="skills-list">
                            {results.job_fit_analysis.missing_skills.map((skill, index) => (
                              <span key={index} className="skill-tag missing">{skill}</span>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="analysis-item">
                        <h5><i className="fas fa-clock"></i> Experience Fit</h5>
                        <p className={`experience-fit ${results.job_fit_analysis.experience_fit === 'Good fit' ? 'good' : 'needs-work'}`}>
                          {results.job_fit_analysis.experience_fit}
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Mock Interview Button */}
                <div className="mock-interview-section">
                  <div className="interview-ready">
                    <h4><i className="fas fa-play-circle"></i> Ready for Mock Interview!</h4>
                    <p>
                      {results.job_fit_analysis
                        ? "Get personalized interview questions based on your resume and the job description."
                        : "Practice with interview questions tailored to your resume and skills."
                      }
                    </p>
                    <button
                      className="mock-interview-btn"
                      onClick={() => {
                        if (jobDescription.trim()) {
                          generateQuestions();
                        } else {
                          setShowMockInterview(true);
                        }
                      }}
                    >
                      <i className="fas fa-microphone"></i>
                      Start Mock Interview
                    </button>
                  </div>
                </div>
              </section>
            )}
          </>
        ) : (
          <MockInterviewSection />
        )}
      </main>
    </div>
  );
};

export default ResumeUpload;