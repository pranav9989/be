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
    const [parsedResumeData, setParsedResumeData] = useState(null);

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

        if (!selectedFile && (!user.skills || user.skills === '[]' || user.skills === '')) {
            setError('Please upload your resume to generate a study plan.');
            return;
        }

        setIsAnalyzing(true);
        setError(null);
        setResult(null);

        try {
            let resumeData = null;

            // STEP 1: Upload the resume if a new one is selected
            if (selectedFile) {
                setUploadProgress(10);
                const progressInterval = setInterval(() => {
                    setUploadProgress(p => p < 85 ? p + Math.random() * 12 : p);
                }, 400);

                const formData = new FormData();
                formData.append('resume', selectedFile);
                formData.append('job_description', 'Gap Analysis Mode');

                const uploadRes = await resumeAPI.upload(formData);
                clearInterval(progressInterval);
                setUploadProgress(100);

                if (!uploadRes.data.success) {
                    throw new Error(uploadRes.data.message || 'Resume upload failed');
                }

                // Store the parsed resume data
                if (uploadRes.data.data) {
                    resumeData = uploadRes.data.data;
                    setParsedResumeData(resumeData);
                    console.log('✅ Resume parsed:', resumeData);
                    console.log('📁 Projects found:', resumeData.projects?.length || 0);
                    console.log('💼 Experience found:', resumeData.experience?.length || 0);
                }

                setTimeout(() => setUploadProgress(0), 1000);
            } else if (parsedResumeData) {
                // Use existing parsed data if no new file uploaded
                resumeData = parsedResumeData;
            }

            // STEP 2: Generate the Gap Analysis
            const response = await api.post('/resume/gap-analysis', {
                job_description: jobDescription
            });

            if (response.data && response.data.success) {
                // Merge gap analysis with parsed resume data
                const mergedResult = {
                    ...response.data.gap_analysis,
                    parsed_resume: resumeData || parsedResumeData || {
                        projects: [],
                        experience: [],
                        experience_years: 0,
                        skills: []
                    }
                };
                console.log('✅ Merged result:', mergedResult);
                setResult(mergedResult);
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

                        {/* Projects and Experience Section */}
                        <div className="resume-details-section mt-6">
                            <h3 className="section-title">
                                <i className="fas fa-file-alt"></i> Resume Details
                            </h3>
                            <div className="details-grid">
                                {/* Projects Card */}
                                <div className="card-glass details-card">
                                    <div className="card-header">
                                        <i className="fas fa-project-diagram text-blue-400"></i>
                                        <h4>Projects Found</h4>
                                        <span className="badge">{result.parsed_resume?.projects?.length || 0}</span>
                                    </div>
                                    <div className="card-body">
                                        {result.parsed_resume?.projects && result.parsed_resume.projects.length > 0 ? (
                                            <div className="items-list">
                                                {result.parsed_resume.projects.map((project, idx) => (
                                                    <div key={idx} className="list-item">
                                                        <div className="item-title">
                                                            <i className="fas fa-code"></i>
                                                            <strong>{project.name}</strong>
                                                        </div>
                                                        {project.tech_stack && project.tech_stack.length > 0 && (
                                                            <div className="tech-tags">
                                                                {project.tech_stack.slice(0, 5).map((tech, tidx) => (
                                                                    <span key={tidx} className="tech-tag">{tech}</span>
                                                                ))}
                                                            </div>
                                                        )}
                                                        <p className="item-description">
                                                            {project.description?.length > 200
                                                                ? project.description.substring(0, 200) + '...'
                                                                : project.description}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <p className="empty-state">No projects detected in resume.</p>
                                        )}
                                    </div>
                                </div>

                                {/* Experience Card */}
                                <div className="card-glass details-card">
                                    <div className="card-header">
                                        <i className="fas fa-briefcase text-green-400"></i>
                                        <h4>Experience & Internships</h4>
                                        <span className="badge">{result.parsed_resume?.experience?.length || 0}</span>
                                    </div>
                                    <div className="card-body">
                                        {result.parsed_resume?.experience && result.parsed_resume.experience.length > 0 ? (
                                            <div className="items-list">
                                                {result.parsed_resume.experience.map((exp, idx) => (
                                                    <div key={idx} className="list-item">
                                                        <div className="item-title">
                                                            <i className="fas fa-building"></i>
                                                            <strong>{exp.title}</strong>
                                                        </div>
                                                        {exp.company && (
                                                            <div className="item-subtitle">
                                                                <i className="fas fa-user-tie"></i> {exp.company}
                                                            </div>
                                                        )}
                                                        <p className="item-description">
                                                            {exp.description?.length > 200
                                                                ? exp.description.substring(0, 200) + '...'
                                                                : exp.description}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <p className="empty-state">No experience detected in resume.</p>
                                        )}
                                    </div>
                                </div>
                                {result.parsed_resume?.certifications && result.parsed_resume.certifications.length > 0 && (
                                    <div className="card-glass certifications-card mt-4">
                                        <div className="card-header">
                                            <i className="fas fa-certificate text-yellow-400"></i>
                                            <h4>Certifications & Courses</h4>
                                            <span className="badge">{result.parsed_resume.certifications.length}</span>
                                        </div>
                                        <div className="certifications-list-detailed">
                                            {result.parsed_resume.certifications.slice(0, 15).map((cert, idx) => (
                                                <div key={idx} className="certification-item-detailed">
                                                    <i className="fas fa-award"></i>
                                                    <span>{cert}</span>
                                                </div>
                                            ))}
                                            {result.parsed_resume.certifications.length > 15 && (
                                                <div className="certification-more-detailed">
                                                    +{result.parsed_resume.certifications.length - 15} more certifications
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Skills Summary Card */}
                            {result.parsed_resume?.skills && result.parsed_resume.skills.length > 0 && (
                                <div className="card-glass skills-summary-card mt-4">
                                    <div className="card-header">
                                        <i className="fas fa-cogs text-purple-400"></i>
                                        <h4>Skills Detected</h4>
                                        <span className="badge">{result.parsed_resume.skills.length}</span>
                                    </div>
                                    <div className="skills-tags-container">
                                        {result.parsed_resume.skills.slice(0, 20).map((skill, idx) => (
                                            <span key={idx} className="skill-tag-detected">{skill}</span>
                                        ))}
                                        {result.parsed_resume.skills.length > 20 && (
                                            <span className="skill-tag-more">+{result.parsed_resume.skills.length - 20} more</span>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Experience Years Summary */}
                            {result.parsed_resume?.experience_years > 0 && (
                                <div className="card-glass experience-summary mt-4">
                                    <div className="card-header">
                                        <i className="fas fa-calendar-alt text-yellow-400"></i>
                                        <h4>Total Experience</h4>
                                    </div>
                                    <div className="exp-years-value">
                                        {result.parsed_resume.experience_years} year{result.parsed_resume.experience_years !== 1 ? 's' : ''}
                                    </div>
                                </div>
                            )}
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