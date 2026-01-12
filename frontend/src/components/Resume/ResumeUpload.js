import React, { useState } from 'react';
import Header from '../Layout/Header';
import MockInterview from '../MockInterview/MockInterview';
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

  const showMessage = (type, text) => {
    setMessage({ type, text });
  };

  return (
    <div className="resume-upload-container">
      <Header 
        user={user} 
        onLogout={onLogout} 
        title="Resume Upload" 
        showBack={true}
      />

      <main className="resume-main">
        {message.text && (
          <div className={`message ${message.type}`}>
            {message.text}
          </div>
        )}

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
            <div className="progress-bar">
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
              {uploading ? 'Uploading...' : 'Upload Resume'}
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
                  onClick={() => setShowMockInterview(true)}
                >
                  <i className="fas fa-microphone"></i>
                  Start Mock Interview
                </button>
              </div>
            </div>
          </section>
        )}

        {/* Mock Interview Interface */}
        {showMockInterview && (
          <MockInterview
            user={user}
            onLogout={onLogout}
            initialJobDescription={jobDescription}
            onBack={() => setShowMockInterview(false)}
          />
        )}
      </main>
    </div>
  );
};

export default ResumeUpload;