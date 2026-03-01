import React, { useState, useRef } from 'react';
import Header from '../Layout/Header';
import MockInterview from '../MockInterview/MockInterview';
import { resumeAPI } from '../../services/api';
import './ResumeUpload.css';

// ─── Mini components ─────────────────────────────────────────────
const SkillTag = ({ skill, variant = '' }) => (
  <span className={`skill-tag ${variant}`}>{skill}</span>
);

const SectionCard = ({ icon, title, children, accent }) => (
  <div className="analysis-card" style={accent ? { '--card-accent': accent } : {}}>
    <h4>
      <i className={icon} />
      {title}
    </h4>
    {children}
  </div>
);

const ScoreRing = ({ pct, label, color, size = 80 }) => {
  const r = (size / 2) - 7;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return (
    <div className="score-ring-wrap" title={`${pct}% ${label}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="score-ring-svg">
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="7" />
        <circle
          cx={size/2} cy={size/2} r={r}
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${size/2} ${size/2})`}
          style={{ transition: 'stroke-dasharray 1s cubic-bezier(.4,0,.2,1)' }}
        />
      </svg>
      <div className="score-ring-inner">
        <strong style={{ color }}>{pct}%</strong>
        <span>{label}</span>
      </div>
    </div>
  );
};

// ─── Main Component ──────────────────────────────────────────────
const ResumeUpload = ({ user, onLogout }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [showMockInterview, setShowMockInterview] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [uploadedUser, setUploadedUser] = useState(null); // user with resume_filename set

  const fileInputRef = useRef(null);

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
    setResults(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) { setError('Please select a resume file'); return; }
    if (!jobDescription.trim()) { setError('Job Description is required to generate targeted questions'); return; }

    const formData = new FormData();
    formData.append('resume', selectedFile);
    formData.append('job_description', jobDescription.trim());

    setError('');
    setUploading(true);
    setUploadProgress(10);

    // Fake progress ticks
    const progressInterval = setInterval(() => {
      setUploadProgress(p => p < 85 ? p + Math.random() * 12 : p);
    }, 400);

    try {
      const response = await resumeAPI.upload(formData);
      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.data.success) {
        throw new Error(response.data.message || 'Upload failed');
      }

      // Patch user object so MockInterview knows the resume is now available
      setUploadedUser({
        ...user,
        resume_filename: `${user.id}_${selectedFile.name}`,
        skills: response.data.data?.skills || [],
        experience_years: response.data.data?.experience_years || 0,
      });

      // Merge job_fit_analysis into the data object for display
      const analysisData = {
        ...response.data.data,
        job_fit_analysis: response.data.job_fit_analysis || response.data.data?.job_fit_analysis,
      };
      setResults(analysisData);

      setTimeout(() => setUploadProgress(0), 1500);
    } catch (err) {
      clearInterval(progressInterval);
      setUploadProgress(0);
      setError(err.response?.data?.message || err.message || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  // When user clicks "Start Mock Interview" from the analysis panel:
  const handleStartMockInterview = () => {
    setShowMockInterview(true);
    // Smooth scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const effectiveUser = uploadedUser || user;

  return (
    <div className="resume-upload-container">
      <Header user={user} onLogout={onLogout} title="Resume Analysis" showBack={true} />

      <main className="resume-main">
        {/* ── Mock Interview overlay ──────────────────────────── */}
        {showMockInterview && (
          <div className="mock-interview-overlay">
            <MockInterview
              user={effectiveUser}
              onLogout={onLogout}
              initialJobDescription={jobDescription}
              onBack={() => setShowMockInterview(false)}
            />
          </div>
        )}

        {!showMockInterview && (
          <>
            {/* ── Page title ───────────────────────────────────── */}
            <div className="ru-page-hero">
              <div className="ru-hero-icon"><i className="fas fa-file-alt" /></div>
              <div className="ru-hero-text">
                <h1>Resume Analysis <span className="ru-hero-amp">&</span> Mock Interview</h1>
                <p>Upload your resume + paste a job description to get AI-powered analysis and a personalised mock interview.</p>
              </div>
            </div>

            {/* ── Upload Card ───────────────────────────────────── */}
            <section className="upload-section">
              <h2><i className="fas fa-cloud-upload-alt" /> Upload Resume</h2>
              <p>PDF, DOC, or DOCX · Max 16 MB</p>

              {/* Drop zone */}
              <div
                className={`upload-area ${dragOver ? 'dragover' : ''} ${selectedFile ? 'has-file' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
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
                    <div className="upload-text">Drag & drop or click to upload</div>
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

              {/* Job Description */}
              <div className="job-description-section">
                <label htmlFor="job-desc" className="job-description-label">
                  <i className="fas fa-briefcase" /> Job Description <span className="required-star">*</span>
                </label>
                <textarea
                  id="job-desc"
                  className="job-description-textarea"
                  value={jobDescription}
                  onChange={e => setJobDescription(e.target.value)}
                  placeholder="Paste the complete job description here. The AI will use this to perform skill-gap analysis and generate targeted interview questions..."
                  rows={5}
                />
                <small className="job-description-help">
                  <i className="fas fa-info-circle" /> More detailed JDs produce more accurate analysis and targeted questions.
                </small>
              </div>

              {/* Upload progress */}
              {uploadProgress > 0 && (
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="upload-error">
                  <i className="fas fa-exclamation-circle" /> {error}
                </div>
              )}

              {/* Upload Button */}
              <button
                className="upload-btn"
                onClick={handleUpload}
                disabled={!selectedFile || !jobDescription.trim() || uploading}
              >
                {uploading ? (
                  <><i className="fas fa-spinner fa-spin" /> Analysing Resume…</>
                ) : results ? (
                  <><i className="fas fa-redo" /> Re-Analyse</>
                ) : (
                  <><i className="fas fa-magic" /> Analyse Resume & Generate Insights</>
                )}
              </button>
            </section>

            {/* ── Analysis Results ──────────────────────────────── */}
            {results && (
              <section className="results-section" id="analysis-results">
                <div className="results-header">
                  <div className="results-header-left">
                    <h3><i className="fas fa-chart-line" /> Analysis Complete</h3>
                    <p>{selectedFile?.name}</p>
                  </div>
                  <button className="start-mock-btn-inline" onClick={handleStartMockInterview}>
                    <i className="fas fa-play" /> Start Mock Interview
                  </button>
                </div>

                <div className="analysis-grid">
                  {/* ── Resume Basics Card ── */}
                  <SectionCard icon="fas fa-id-card" title="Resume Basics">
                    <div className="analysis-item">
                      <h5><i className="fas fa-calendar-alt" /> Experience</h5>
                      <p className="experience-text">
                        {results.experience_years || 0} year{(results.experience_years || 0) !== 1 ? 's' : ''}
                        {results.experience_years === 0 && ' (fresher / not detected)'}
                      </p>
                    </div>

                    {results.skills?.length > 0 && (
                      <div className="analysis-item">
                        <h5><i className="fas fa-cogs" /> Skills ({results.skills.length})</h5>
                        <div className="skills-list">
                          {results.skills.map((s, i) => <SkillTag key={i} skill={s} />)}
                        </div>
                      </div>
                    )}

                    {results.certifications?.length > 0 && (
                      <div className="analysis-item">
                        <h5><i className="fas fa-certificate" /> Certifications</h5>
                        <div className="certifications-list">
                          {results.certifications.filter(c => c.length > 5).map((c, i) => (
                            <span key={i} className="cert-tag">{c}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {results.internships?.length > 0 && (
                      <div className="analysis-item">
                        <h5><i className="fas fa-briefcase" /> Work / Internships</h5>
                        <ul className="internships-list">
                          {results.internships.map((intship, i) => <li key={i}>{intship}</li>)}
                        </ul>
                      </div>
                    )}
                  </SectionCard>

                  {/* ── Projects Card ── */}
                  {results.projects?.length > 0 && (
                    <SectionCard icon="fas fa-project-diagram" title="Key Projects">
                      <div className="projects-grid">
                        {results.projects.map((project, i) => {
                          const name = typeof project === 'object' ? project.name : (project.split('\n')[0] || project);
                          const techs = typeof project === 'object' ? (project.tech_stack || []) : [];
                          const desc = typeof project === 'object' ? '' : project.split('\n').slice(1).join(' ');
                          return (
                            <div key={i} className="project-card">
                              <div className="project-name">{name}</div>
                              {techs.length > 0 && (
                                <div className="project-tech">
                                  {techs.map((t, j) => <span key={j} className="tech-badge">{t}</span>)}
                                </div>
                              )}
                              {desc && <p className="project-description">{desc.slice(0, 180)}{desc.length > 180 ? '…' : ''}</p>}
                            </div>
                          );
                        })}
                      </div>
                    </SectionCard>
                  )}

                  {/* ── Job Fit Analysis Card ── */}
                  {results.job_fit_analysis && (
                    <SectionCard icon="fas fa-bullseye" title="Job Fit Analysis">
                      <div className="match-score">
                        <ScoreRing
                          pct={Math.round(results.job_fit_analysis.match_percentage || 0)}
                          label="Match"
                          color={
                            results.job_fit_analysis.match_percentage >= 70 ? '#22c55e'
                            : results.job_fit_analysis.match_percentage >= 45 ? '#f59e0b'
                            : '#ef4444'
                          }
                        />
                        {results.job_fit_analysis.semantic_similarity > 0 && (
                          <ScoreRing
                            pct={Math.round(results.job_fit_analysis.semantic_similarity * 100)}
                            label="Semantic"
                            color="#9E95F5"
                          />
                        )}
                        <div className="fit-meta">
                          <span className={`gap-severity severity-${results.job_fit_analysis.gap_severity?.toLowerCase()}`}>
                            {results.job_fit_analysis.gap_severity} Skill Gap
                          </span>
                          <span className={`experience-fit ${results.job_fit_analysis.experience_fit === 'Good fit' ? 'good' : 'needs-work'}`}>
                            <i className="fas fa-user-clock" /> {results.job_fit_analysis.experience_fit}
                          </span>
                        </div>
                      </div>

                      {results.job_fit_analysis.matching_skills?.length > 0 && (
                        <div className="analysis-item">
                          <h5><i className="fas fa-check-circle" /> Matching Skills</h5>
                          <div className="skills-list">
                            {results.job_fit_analysis.matching_skills.map((s, i) => (
                              <SkillTag key={i} skill={s} variant="match" />
                            ))}
                          </div>
                        </div>
                      )}

                      {results.job_fit_analysis.missing_skills?.length > 0 && (
                        <div className="analysis-item">
                          <h5><i className="fas fa-exclamation-triangle" /> Skills to Develop</h5>
                          <div className="skills-list">
                            {results.job_fit_analysis.missing_skills.map((s, i) => (
                              <SkillTag key={i} skill={s} variant="missing" />
                            ))}
                          </div>
                        </div>
                      )}

                      {results.job_fit_analysis.jd_skills_found?.length > 0 && (
                        <div className="analysis-item">
                          <h5><i className="fas fa-search" /> Skills Found in JD</h5>
                          <div className="skills-list">
                            {results.job_fit_analysis.jd_skills_found.map((s, i) => (
                              <SkillTag key={i} skill={s} variant="" />
                            ))}
                          </div>
                        </div>
                      )}
                    </SectionCard>
                  )}
                </div>

                {/* ── START INTERVIEW CTA ── */}
                <div className="mock-interview-section">
                  <div className="interview-ready">
                    <div className="interview-ready-icon">
                      <i className="fas fa-robot" />
                    </div>
                    <div className="interview-ready-content">
                      <h4><i className="fas fa-star" /> Ready for Your AI Mock Interview?</h4>
                      <p>
                        Our Gemini-powered interviewer will ask{' '}
                        <strong>custom questions based on your resume and the job description</strong>.
                        Every answer is scored in real time with detailed AI feedback.
                      </p>
                      <div className="interview-features">
                        <span><i className="fas fa-brain" /> Gemini AI Evaluation</span>
                        <span><i className="fas fa-chart-bar" /> Per-Question Scoring</span>
                        <span><i className="fas fa-lightbulb" /> Model Answers</span>
                        <span><i className="fas fa-clock" /> Timed Session</span>
                      </div>
                    </div>
                    <button className="mock-interview-btn" onClick={handleStartMockInterview}>
                      <i className="fas fa-play-circle" /> Start Mock Interview
                    </button>
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
};

export default ResumeUpload;