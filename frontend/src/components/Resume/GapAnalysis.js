import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import api, { resumeAPI } from '../../services/api';
import Header from '../Layout/Header';
import './GapAnalysis.css';
import '../Resume/ResumeUpload.css';

const GapAnalysis = ({ user, onLogout }) => {
    const navigate = useNavigate();
    
    // File Upload State
    const [selectedFile, setSelectedFile] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const fileInputRef = useRef(null);

    // Analysis State
    const [jobDescription, setJobDescription] = useState('');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const isValidFile = (file) => {
        const allowed = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ];
        return allowed.includes(file.type) && file.size <= 16 * 1024 * 1024;
    };

    const handleFileSelect = (file) => {
        if (!file) return;
        if (!isValidFile(file)) {
            setError('Please select a valid PDF, DOC, or DOCX file (max 16MB)');
            return;
        }
        setSelectedFile(file);
        setError('');
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleFileSelect(file);
    };

    const handleAnalyze = async () => {
        if (!jobDescription.trim()) {
            setError('Please paste a job description first.');
            return;
        }

        // If user hasn't selected a new file, and doesn't have an existing resume/skills, throw error.
        if (!selectedFile && (!user.skills || user.skills === '[]' || user.skills === '')) {
            setError('Please upload your resume to generate a study plan.');
            return;
        }

        setIsAnalyzing(true);
        setError(null);
        setResult(null);

        try {
            // STEP 1: Upload the resume if a new one is selected
            if (selectedFile) {
                setUploadProgress(10);
                const progressInterval = setInterval(() => {
                    setUploadProgress(p => p < 85 ? p + Math.random() * 12 : p);
                }, 400);

                const formData = new FormData();
                formData.append('resume', selectedFile);
                formData.append('job_description', 'Gap Analysis Mode'); // Just a placeholder for the upload API
                
                const uploadRes = await resumeAPI.upload(formData);
                clearInterval(progressInterval);
                setUploadProgress(100);
                
                if (!uploadRes.data.success) {
                    throw new Error(uploadRes.data.message || 'Resume upload failed');
                }
                
                setTimeout(() => setUploadProgress(0), 1000);
            }

            // STEP 2: Generate the Gap Analysis (which now uses the updated DB skills)
            const response = await api.post('/resume/gap-analysis', {
                job_description: jobDescription
            });

            if (response.data && response.data.success) {
                setResult(response.data.gap_analysis);
            } else {
                setError(response.data.error || 'Failed to analyze gap.');
            }
        } catch (err) {
            console.error('Gap analysis error:', err);
            setUploadProgress(0);
            setError(err.response?.data?.message || err.response?.data?.error || err.message || 'An error occurred during analysis.');
        } finally {
            setIsAnalyzing(false);
        }
    };

    const getScoreColor = (score) => {
        if (score >= 80) return 'var(--color-success)';
        if (score >= 50) return 'var(--color-warning)';
        return 'var(--color-error)';
    };

    return (
        <div className="page-container">
            <Header user={user} onLogout={onLogout} title="Resume vs Reality" />
            
            <main className="main-content gap-analysis-main">
                <div className="gap-intro">
                    <h1 className="gradient-text">Resume vs Reality</h1>
                    <p>Find out exactly what's missing between your current skills and your dream job.</p>
                </div>

                {!result ? (
                    <div className="card-glass gap-input-card fadeIn">
                        <h2><i className="fas fa-bullseye"></i> Target Job Description</h2>
                        <p className="text-secondary mb-4">
                            Upload your latest resume and paste the full job description below. Our AI will compare them to reveal your skill gaps.
                        </p>
                        
                        {error && (
                            <div className="message error mb-4">
                                <i className="fas fa-exclamation-circle"></i> {error}
                            </div>
                        )}

                        {/* Drop zone */}
                        {!user.skills || user.skills === '[]' || user.skills === '' || selectedFile ? (
                            <>
                                <div
                                    className={`upload-area ${dragOver ? 'dragover' : ''} ${selectedFile ? 'has-file' : ''}`}
                                    onClick={() => fileInputRef.current?.click()}
                                    onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                                    onDragLeave={() => setDragOver(false)}
                                    onDrop={handleDrop}
                                    style={{ marginBottom: '1.5rem', marginTop: '1rem' }}
                                >
                                    {selectedFile ? (
                                    <>
                                        <div className="upload-file-icon">
                                            <i className={selectedFile.name.endsWith('.pdf') ? 'fas fa-file-pdf' : 'fas fa-file-word'} />
                                        </div>
                                        <div className="upload-text">{selectedFile.name}</div>
                                        <div className="upload-subtext">
                                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB ·{' '}
                                        <span className="upload-change">Click to change</span>
                                        </div>
                                    </>
                                    ) : (
                                    <>
                                        <i className="fas fa-arrow-up-from-bracket upload-icon" />
                                        <div className="upload-text">Drag & drop your Resume</div>
                                        <div className="upload-subtext">PDF, DOC, DOCX up to 16 MB</div>
                                    </>
                                    )}
                                </div>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    style={{ display: 'none' }}
                                    accept=".pdf,.doc,.docx"
                                    onChange={e => handleFileSelect(e.target.files?.[0])}
                                />
                            </>
                        ) : (
                            <div className="message success mb-4" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <div><i className="fas fa-file-invoice"></i> We have your previously saved resume on file.</div>
                                <button className="btn btn-secondary btn-sm" onClick={() => fileInputRef.current?.click()}>
                                    Upload New Resume
                                </button>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    style={{ display: 'none' }}
                                    accept=".pdf,.doc,.docx"
                                    onChange={e => handleFileSelect(e.target.files?.[0])}
                                />
                            </div>
                        )}

                        <textarea
                            className="form-textarea gap-textarea"
                            value={jobDescription}
                            onChange={(e) => setJobDescription(e.target.value)}
                            placeholder="Paste Job Description here..."
                            disabled={isAnalyzing}
                        ></textarea>

                        {/* Upload progress */}
                        {uploadProgress > 0 && (
                            <div className="progress-bar mb-3">
                                <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
                            </div>
                        )}


                        <div className="gap-action-row">
                            <button 
                                className="btn btn-secondary"
                                onClick={() => navigate('/dashboard')}
                                disabled={isAnalyzing}
                            >
                                Cancel
                            </button>
                            <button 
                                className="btn btn-primary btn-generate"
                                onClick={handleAnalyze}
                                disabled={isAnalyzing || !jobDescription.trim()}
                            >
                                {isAnalyzing ? (
                                    <>
                                        <i className="fas fa-spinner fa-spin"></i> Analyzing Gap...
                                    </>
                                ) : (
                                    <>
                                        <i className="fas fa-magic"></i> Generate Study Plan
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                ) : (
                    <div className="gap-results slideUp">
                        <div className="gap-actions-top">
                            <button className="btn btn-secondary" onClick={() => setResult(null)}>
                                <i className="fas fa-arrow-left"></i> Analyze Another Job
                            </button>
                        </div>

                        <div className="gap-dashboard">
                            {/* Score Card */}
                            <div className="card-glass score-card">
                                <h3>Match Score</h3>
                                <div className="score-circle-container">
                                    <div 
                                        className="score-circle"
                                        style={{ '--score-color': getScoreColor(result.match_score) }}
                                    >
                                        <span className="score-number">{result.match_score}%</span>
                                    </div>
                                </div>
                                <p className="score-desc">
                                    {result.match_score >= 80 ? "You are a strong fit! Focus on interview prep." : 
                                     result.match_score >= 50 ? "You have the basics, but need to bridge some gaps." : 
                                     "Significant gaps detected. A study plan is highly recommended."}
                                </p>
                            </div>

                            {/* Missing Skills Card */}
                            <div className="card-glass skills-card">
                                <h3>Missing Key Skills</h3>
                                {result.missing_skills && result.missing_skills.length > 0 ? (
                                    <div className="skills-tags">
                                        {result.missing_skills.map((skill, idx) => (
                                            <span key={idx} className="skill-tag missing">
                                                <i className="fas fa-times-circle"></i> {skill}
                                            </span>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="message success">
                                        <i className="fas fa-check-circle"></i> No major skills missing!
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Study Plan Timeline */}
                        {result.study_plan && result.study_plan.length > 0 && (
                            <div className="card-glass study-plan-card mt-4">
                                <h3 className="mb-4"><i className="fas fa-calendar-check icon-gold"></i> Your 2-Week Study Roadmap</h3>
                                <div className="timeline">
                                    {result.study_plan.map((day, idx) => (
                                        <div key={idx} className="timeline-item">
                                            <div className="timeline-marker">Day {day.day}</div>
                                            <div className="timeline-content">
                                                <h4>{day.topic}</h4>
                                                <p>{day.description}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </main>
        </div>
    );
};

export default GapAnalysis;
