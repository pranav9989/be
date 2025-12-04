import React, { useState } from 'react';
import Header from '../Layout/Header';
import { resumeAPI } from '../../services/api';
import './ResumeUpload.css';

const ResumeUpload = ({ user, onLogout }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [results, setResults] = useState(null);
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

    setUploading(true);
    setUploadProgress(0);

    try {
      const response = await resumeAPI.upload(formData);
      setResults(response.data.data);
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

        {results && (
          <section className="results-section">
            <div className="results-header">
              <h3><i className="fas fa-check-circle"></i> Resume Analysis Complete!</h3>
              <p>Here's what we found in your resume:</p>
            </div>

            <div className="results-grid">
              <div className="result-card">
                <h4><i className="fas fa-cogs"></i> Extracted Skills</h4>
                <div className="skills-list">
                  {results.skills.map((skill, index) => (
                    <span key={index} className="skill-tag">{skill}</span>
                  ))}
                </div>
              </div>

              <div className="result-card">
                <h4><i className="fas fa-calendar-alt"></i> Experience</h4>
                <p>{results.experience_years} year{results.experience_years !== 1 ? 's' : ''}</p>
              </div>

              <div className="result-card">
                <h4><i className="fas fa-lightbulb"></i> Next Steps</h4>
                <p>Your profile has been updated! You can now:</p>
                <ul>
                  <li><a href="/technical-chat">Start technical practice</a></li>
                  <li><a href="/hr-interview">Practice HR questions</a></li>
                </ul>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};

export default ResumeUpload;