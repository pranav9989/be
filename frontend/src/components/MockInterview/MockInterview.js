/**
 * MockInterview.js — Complete AI-powered mock interview experience
 *
 * Features:
 * - Gemini-powered question generation (resume + JD aware)
 * - Per-question AI evaluation with grades, strengths, improvements, ideal answer
 * - Real-time auto-evaluate on "Next" press (non-blocking)
 * - Animated question cards with difficulty badges
 * - Session summary with type-wise performance breakdown
 * - Timer, progress bar, question navigator
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { resumeAPI, hrAPI } from '../../services/api';
import './MockInterview.css';

// ─── Helpers ──────────────────────────────────────────────────────────────────
const GRADE_CONFIG = {
  A: { color: '#22c55e', bg: 'rgba(34,197,94,0.12)', label: 'Excellent' },
  B: { color: '#84cc16', bg: 'rgba(132,204,22,0.12)', label: 'Good' },
  C: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', label: 'Average' },
  D: { color: '#f97316', bg: 'rgba(249,115,22,0.12)', label: 'Below Average' },
  F: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', label: 'Poor' },
};

const TYPE_ICONS = {
  'technical':       'fas fa-code',
  'behavioral':      'fas fa-users',
  'project-based':   'fas fa-project-diagram',
  'situational':     'fas fa-lightbulb',
  'system-design':   'fas fa-server',
};

const DIFF_CONFIG = {
  easy:   { color: '#22c55e', label: 'Easy' },
  medium: { color: '#f59e0b', label: 'Medium' },
  hard:   { color: '#ef4444', label: 'Hard' },
};

const renderItem = (item) => {
  if (!item) return '';
  if (typeof item === 'string') return item;
  if (typeof item === 'object') return item.missing || item.suggestion || item.keyword || Object.values(item).join(': ');
  return String(item);
};

function useTimer(running) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setSeconds(s => s + 1), 1000);
    return () => clearInterval(id);
  }, [running]);
  const format = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  return { seconds, formatted: format(seconds) };
}

// ─── STAGE: Setup ─────────────────────────────────────────────────────────────
function SetupStage({ initialJD, onStart }) {
  const [jd, setJd] = useState(initialJD || '');
  const [count, setCount] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleStart = async () => {
    if (!jd.trim()) { setError('Job description is required to generate targeted questions.'); return; }
    setError('');
    setLoading(true);
    try {
      const seed = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      const res = await resumeAPI.generateQuestions({
        job_description: jd.trim(),
        question_count: count,
        variation_seed: seed,
      });
      if (!res.data.success) throw new Error(res.data.error || 'Failed to generate questions');
      onStart(res.data.questions, jd.trim());
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to generate questions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mi-setup">
      <div className="mi-setup-hero">
        <div className="mi-setup-icon">
          <i className="fas fa-brain" />
        </div>
        <h2>AI Mock Interview</h2>
        <p>Our Gemini-powered AI will generate custom questions from your resume and score every answer in real time.</p>
      </div>

      <div className="mi-setup-card">
        <div className="mi-form-row">
          <label className="mi-label">
            <i className="fas fa-briefcase" /> Job Description <span className="mi-required">*</span>
          </label>
          <textarea
            className="mi-textarea"
            value={jd}
            onChange={e => setJd(e.target.value)}
            placeholder="Paste the complete job description here. The AI will generate questions specifically tailored to this role and your resume..."
            rows={6}
          />
          <span className="mi-help">
            <i className="fas fa-info-circle" /> More detailed JD = more targeted questions
          </span>
        </div>

        <div className="mi-form-row mi-count-row">
          <label className="mi-label" htmlFor="mi-count-input">
            <i className="fas fa-list-ol" /> Number of Questions
          </label>
          <div className="mi-count-input-wrap">
            <input
              id="mi-count-input"
              type="number"
              className="mi-count-input"
              min={1}
              max={30}
              value={count}
              onChange={e => {
                const v = parseInt(e.target.value, 10);
                if (!isNaN(v)) setCount(Math.max(1, Math.min(30, v)));
              }}
            />
            <span className="mi-count-hint">questions (1 – 30)</span>
          </div>
          <small className="mi-help">
            <i className="fas fa-info-circle" /> More questions → deeper practice. Recommended: 5–15 for a focused session.
          </small>
        </div>

        <div className="mi-features-row">
          {[
            { icon: 'fas fa-robot', text: 'AI-graded answers' },
            { icon: 'fas fa-chart-bar', text: 'Performance analytics' },
            { icon: 'fas fa-lightbulb', text: 'Per-question feedback' },
            { icon: 'fas fa-clock', text: 'Timed session' },
          ].map((f, i) => (
            <div key={i} className="mi-feature-chip">
              <i className={f.icon} /> {f.text}
            </div>
          ))}
        </div>

        {error && <div className="mi-error"><i className="fas fa-exclamation-circle" /> {error}</div>}

        <button className="mi-start-btn" onClick={handleStart} disabled={loading || !jd.trim()}>
          {loading ? (
            <><i className="fas fa-spinner fa-spin" /> Generating Questions with AI...</>
          ) : (
            <><i className="fas fa-play" /> Start Mock Interview</>
          )}
        </button>
      </div>
    </div>
  );
}

// ─── STAGE: Interview ─────────────────────────────────────────────────────────
function InterviewStage({ questions, jobDescription, onFinish }) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [answers, setAnswers] = useState({});
  const [evaluations, setEvaluations] = useState({});
  const [evaluating, setEvaluating] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const textareaRef = useRef(null);
  const { seconds, formatted: timerStr } = useTimer(true);

  const current = questions[currentIdx];
  const currentAnswer = answers[currentIdx] || '';
  const currentEval = evaluations[currentIdx];
  const progress = ((currentIdx + 1) / questions.length) * 100;
  const answeredCount = Object.values(answers).filter(a => a?.trim()).length;

  // Focus textarea when question changes
  useEffect(() => {
    if (textareaRef.current) textareaRef.current.focus();
    setShowHint(false);
  }, [currentIdx]);

  const handleAnswerChange = useCallback((val) => {
    setAnswers(prev => ({ ...prev, [currentIdx]: val }));
    // Clear stale evaluation when answer changes significantly
    if (evaluations[currentIdx] && val.length < (answers[currentIdx] || '').length - 20) {
      setEvaluations(prev => { const n = {...prev}; delete n[currentIdx]; return n; });
    }
  }, [currentIdx, evaluations, answers]);

  const evaluateCurrent = useCallback(async (questionObj, answer, idx) => {
    if (!answer?.trim() || evaluations[idx]) return;
    setEvaluating(true);
    try {
      const res = await resumeAPI.evaluateAnswer({
        question: questionObj,
        answer: answer,
        job_description: jobDescription,
      });
      if (res.data.success) {
        setEvaluations(prev => ({ ...prev, [idx]: res.data.evaluation }));
      }
    } catch (err) {
      console.warn('[MockInterview] Evaluation failed:', err.message);
    } finally {
      setEvaluating(false);
    }
  }, [jobDescription, evaluations]);

  const goTo = async (idx, saveEval = true) => {
    // Evaluate current answer before moving (non-blocking)
    if (saveEval && currentAnswer?.trim() && !evaluations[currentIdx]) {
      evaluateCurrent(current, currentAnswer, currentIdx);
    }
    setCurrentIdx(idx);
    setNavOpen(false);
  };

  const handleNext = () => {
    if (currentIdx < questions.length - 1) {
      goTo(currentIdx + 1);
    } else {
      // Last question — finish
      handleFinish();
    }
  };

  const handleFinish = async () => {
    setEvaluating(true);
    const finalEvaluations = { ...evaluations };
    
    // Find all questions without an evaluation (both answered and skipped)
    const missingIndices = questions.map((_, i) => i).filter(i => !finalEvaluations[i]);
    
    await Promise.all(
      missingIndices.map(async (idx) => {
        try {
          const res = await resumeAPI.evaluateAnswer({
            question: questions[idx],
            answer: answers[idx] || '', // Pass empty string for skipped
            job_description: jobDescription,
          });
          if (res.data.success) {
            finalEvaluations[idx] = res.data.evaluation;
          }
        } catch (err) {
          console.warn(`[MockInterview] Eval failed for q${idx}:`, err.message);
        }
      })
    );
    
    setEvaluations(finalEvaluations);
    setEvaluating(false);

    onFinish({
      questions,
      answers,
      evaluations: finalEvaluations,
      jobDescription,
      durationSeconds: seconds,
    });
  };

  return (
    <div className="mi-interview">
      {/* Header bar */}
      <div className="mi-interview-header">
        <div className="mi-header-left">
          <button className="mi-nav-toggle" onClick={() => setNavOpen(v => !v)} title="Question Navigator">
            <i className="fas fa-th-large" />
          </button>
          <div className="mi-progress-stats">
            <span className="mi-q-counter">Q{currentIdx + 1} <em>of {questions.length}</em></span>
            <span className="mi-answered-stat">
              <i className="fas fa-check-circle" /> {answeredCount} answered
            </span>
          </div>
        </div>
        <div className="mi-header-center">
          <div className="mi-progress-bar-wrap">
            <div className="mi-progress-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
        <div className="mi-header-right">
          <div className="mi-timer">
            <i className="fas fa-clock" /> {timerStr}
          </div>
          <button className="mi-finish-early-btn" onClick={handleFinish}>
            <i className="fas fa-flag-checkered" /> Finish
          </button>
        </div>
      </div>

      {/* Question Navigator Drawer */}
      {navOpen && (
        <div className="mi-nav-drawer">
          <div className="mi-nav-grid">
            {questions.map((q, i) => (
              <button
                key={i}
                className={`mi-nav-dot ${i === currentIdx ? 'current' : ''} ${answers[i]?.trim() ? 'answered' : ''} ${evaluations[i] ? `graded grade-${evaluations[i].grade}` : ''}`}
                onClick={() => goTo(i)}
                title={`Q${i + 1}: ${q.question?.slice(0, 60)}...`}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <div className="mi-nav-legend">
            <span><span className="mi-nav-dot answered sm" /> Answered</span>
            <span><span className="mi-nav-dot sm" /> Unanswered</span>
            <span><span className="mi-nav-dot current sm" /> Current</span>
          </div>
        </div>
      )}

      {/* Question card */}
      <div className="mi-question-area" key={currentIdx}>
        <div className="mi-question-meta">
          <span className="mi-type-badge">
            <i className={TYPE_ICONS[current.type] || 'fas fa-question'} />
            {current.type?.replace(/-/g, ' ')}
          </span>
          <span
            className="mi-diff-badge"
            style={{
              color: DIFF_CONFIG[current.difficulty]?.color || '#f59e0b',
              background: `${DIFF_CONFIG[current.difficulty]?.color || '#f59e0b'}18`,
            }}
          >
            {DIFF_CONFIG[current.difficulty]?.label || 'Medium'}
          </span>
          {current.focus_area && (
            <span className="mi-focus-badge">
              <i className="fas fa-crosshairs" /> {current.focus_area}
            </span>
          )}
        </div>

        <div className="mi-question-text">
          <span className="mi-q-num">Q{currentIdx + 1}.</span> {current.question}
        </div>

        {current.hint && (
          <div className="mi-hint-row">
            <button className="mi-hint-toggle" onClick={() => setShowHint(v => !v)}>
              <i className={`fas fa-${showHint ? 'eye-slash' : 'lightbulb'}`} />
              {showHint ? 'Hide hint' : 'Show hint'}
            </button>
            {showHint && (
              <div className="mi-hint-text">
                <i className="fas fa-info-circle" /> {current.hint}
              </div>
            )}
          </div>
        )}

        <div className="mi-answer-section">
          <label className="mi-answer-label">
            <i className="fas fa-pen-alt" /> Your Answer
            <span className="mi-word-count">{currentAnswer.trim().split(/\s+/).filter(Boolean).length} words</span>
          </label>
          <textarea
            ref={textareaRef}
            className="mi-answer-textarea"
            value={currentAnswer}
            onChange={e => handleAnswerChange(e.target.value)}
            placeholder="Type your answer here. Be specific, use examples, and cover key concepts. The AI will evaluate your response when you proceed..."
            rows={8}
          />
        </div>

        {/* Live evaluation panel (shown if answer was evaluated) */}
        {currentEval && (
          <EvalPanel evaluation={currentEval} />
        )}

        {/* Evaluating indicator */}
        {evaluating && (
          <div className="mi-evaluating-badge">
            <i className="fas fa-spinner fa-spin" /> AI is evaluating your answer...
          </div>
        )}
      </div>

      {/* Navigation footer */}
      <div className="mi-nav-footer">
        <button
          className="mi-nav-btn prev"
          onClick={() => goTo(currentIdx - 1, false)}
          disabled={currentIdx === 0}
        >
          <i className="fas fa-arrow-left" /> Previous
        </button>

        <div className="mi-nav-center">
          {current.expected_keywords?.length > 0 && (
            <div className="mi-kw-preview">
              <span className="mi-kw-label">Key concepts:</span>
              {current.expected_keywords.slice(0, 4).map((kw, i) => (
                <span key={i} className={`mi-kw-tag ${currentAnswer.toLowerCase().includes(kw.toLowerCase()) ? 'covered' : ''}`}>
                  {kw}
                </span>
              ))}
            </div>
          )}
        </div>

        <button
          className="mi-nav-btn next"
          onClick={handleNext}
        >
          {currentIdx === questions.length - 1 ? (
            <><i className="fas fa-flag-checkered" /> Finish Interview</>
          ) : (
            <>Next <i className="fas fa-arrow-right" /></>
          )}
        </button>
      </div>
    </div>
  );
}

// ─── Evaluation Panel ─────────────────────────────────────────────────────────
function EvalPanel({ evaluation }) {
  const [expanded, setExpanded] = useState(false);
  const grade = evaluation.grade || 'C';
  const cfg = GRADE_CONFIG[grade] || GRADE_CONFIG.C;

  return (
    <div className="mi-eval-panel" style={{ '--eval-color': cfg.color, '--eval-bg': cfg.bg }}>
      <div className="mi-eval-header" onClick={() => setExpanded(v => !v)}>
        <div className="mi-eval-grade" style={{ background: cfg.bg, color: cfg.color }}>
          {grade}
        </div>
        <div className="mi-eval-meta">
          <strong>{cfg.label}</strong>
          <span>{evaluation.score}/100 · {evaluation.verdict}</span>
        </div>
        <div className="mi-eval-score-bar">
          <div className="mi-eval-score-fill" style={{ width: `${evaluation.score}%`, background: cfg.color }} />
        </div>
        <i className={`fas fa-chevron-${expanded ? 'up' : 'down'} mi-eval-toggle`} />
      </div>

      {expanded && (
        <div className="mi-eval-body">
          {evaluation.strengths?.length > 0 && (
            <div className="mi-eval-section">
              <h6><i className="fas fa-check-circle" style={{ color: '#22c55e' }} /> Strengths</h6>
              <ul>
                {evaluation.strengths.map((s, i) => <li key={i}>{renderItem(s)}</li>)}
              </ul>
            </div>
          )}

          {evaluation.improvements?.length > 0 && (
            <div className="mi-eval-section">
              <h6><i className="fas fa-arrow-up-right-dots" style={{ color: '#f59e0b' }} /> Areas to Improve</h6>
              <ul className="mi-improvements">
                {evaluation.improvements.map((s, i) => <li key={i}>{renderItem(s)}</li>)}
              </ul>
            </div>
          )}

          {evaluation.ideal_answer && (
            <div className="mi-eval-section mi-ideal-answer">
              <h6><i className="fas fa-star" style={{ color: '#D4A853' }} /> Model Answer</h6>
              <p>{evaluation.ideal_answer}</p>
            </div>
          )}

          {(evaluation.keyword_coverage?.length > 0 || evaluation.missing_keywords?.length > 0) && (
            <div className="mi-eval-section">
              <h6><i className="fas fa-tags" /> Keyword Coverage</h6>
              <div className="mi-kw-coverage">
                {evaluation.keyword_coverage?.map((kw, i) => (
                  <span key={i} className="mi-kw-covered"><i className="fas fa-check" /> {renderItem(kw)}</span>
                ))}
                {evaluation.missing_keywords?.map((kw, i) => (
                  <span key={i} className="mi-kw-missing"><i className="fas fa-times" /> {renderItem(kw)}</span>
                ))}
              </div>
            </div>
          )}

          {evaluation.score_breakdown && (
            <div className="mi-eval-section">
              <h6><i className="fas fa-chart-pie" /> Score Breakdown</h6>
              <div className="mi-score-breakdown">
                {Object.entries(evaluation.score_breakdown).map(([key, val]) => (
                  <div key={key} className="mi-breakdown-row">
                    <span className="mi-breakdown-label">{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                    <div className="mi-breakdown-bar-wrap">
                      <div className="mi-breakdown-bar" style={{ width: `${(val / { relevance: 40, technical: 30, structure: 20, keywords: 10 }[key] || 1) * 100}%`, background: cfg.color }} />
                    </div>
                    <span className="mi-breakdown-val">{val}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── STAGE: Results ───────────────────────────────────────────────────────────
function ResultsStage({ data, onRetry, onNewResume }) {
  const { questions, answers, evaluations, jobDescription, durationSeconds } = data;
  const [summary, setSummary] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [expandedQ, setExpandedQ] = useState(null);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const res = await resumeAPI.getSessionSummary({
          questions,
          answers: Object.fromEntries(
            Object.entries(answers).map(([k, v]) => [String(k), v])
          ),
          evaluations: Object.fromEntries(
            Object.entries(evaluations).map(([k, v]) => [String(k), v])
          ),
          job_description: jobDescription,
          duration_minutes: Math.round(durationSeconds / 60),
        });
        if (res.data.success) setSummary(res.data.summary);
      } catch (err) {
        console.warn('[MockInterview] Summary generation failed:', err.message);
      } finally {
        setLoadingSummary(false);
      }
    };
    fetchSummary();
  }, [answers, durationSeconds, evaluations, jobDescription, questions]);

  const overallGrade = summary?.overall_grade || 'C';
  const gradeConfig = GRADE_CONFIG[overallGrade] || GRADE_CONFIG.C;
  const avgScore = summary?.avg_score || 0;
  const minutes = Math.floor(durationSeconds / 60);
  const secs = durationSeconds % 60;

  return (
    <div className="mi-results">
      {/* Hero score card */}
      <div className="mi-results-hero">
        <div className="mi-results-grade" style={{ '--grade-color': gradeConfig.color, '--grade-bg': gradeConfig.bg }}>
          <div className="mi-grade-ring">
            <svg viewBox="0 0 120 120" className="mi-grade-svg">
              <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" />
              <circle
                cx="60" cy="60" r="52"
                fill="none"
                stroke={gradeConfig.color}
                strokeWidth="8"
                strokeDasharray={`${(avgScore / 100) * 326.7} 326.7`}
                strokeLinecap="round"
                transform="rotate(-90 60 60)"
              />
            </svg>
            <div className="mi-grade-center">
              <span className="mi-grade-letter" style={{ color: gradeConfig.color }}>{overallGrade}</span>
              <span className="mi-grade-score">{avgScore}%</span>
            </div>
          </div>
          <div className="mi-grade-info">
            <h2>Interview Complete!</h2>
            <p className="mi-grade-verdict" style={{ color: gradeConfig.color }}>{gradeConfig.label} Performance</p>
            <div className="mi-stats-row">
              <div className="mi-stat">
                <strong>{summary?.questions_answered ?? Object.values(answers).filter(a => a?.trim()).length}</strong>
                <span>Answered</span>
              </div>
              <div className="mi-stat">
                <strong>{questions.length}</strong>
                <span>Total</span>
              </div>
              <div className="mi-stat">
                <strong>{minutes}:{String(secs).padStart(2, '0')}</strong>
                <span>Duration</span>
              </div>
              <div className="mi-stat">
                <strong>{summary?.completion_rate ?? '—'}%</strong>
                <span>Completion</span>
              </div>
            </div>
          </div>
        </div>

        {summary?.recommendation && (
          <div className="mi-recommendation">
            <i className="fas fa-lightbulb" />
            <p>{summary.recommendation}</p>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="mi-results-tabs">
        {['overview', 'questions', 'insights'].map(tab => (
          <button
            key={tab}
            className={`mi-tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && summary && (
        <div className="mi-tab-content">
          <div className="mi-overview-grid">
            {/* Strengths */}
            <div className="mi-overview-card">
              <h4><i className="fas fa-check-double" style={{ color: '#22c55e' }} /> Key Strengths</h4>
              {summary.top_strengths?.length > 0 ? (
                <ul className="mi-overview-list strengths">
                  {summary.top_strengths.map((s, i) => <li key={i}>{renderItem(s)}</li>)}
                </ul>
              ) : <p className="mi-empty">No strengths recorded</p>}
            </div>

            {/* Improvements */}
            <div className="mi-overview-card">
              <h4><i className="fas fa-arrow-trend-up" style={{ color: '#f59e0b' }} /> Focus Areas</h4>
              {summary.top_improvements?.length > 0 ? (
                <ul className="mi-overview-list improvements">
                  {summary.top_improvements.map((s, i) => <li key={i}>{renderItem(s)}</li>)}
                </ul>
              ) : <p className="mi-empty">No improvements recorded</p>}
            </div>

            {/* Type performance */}
            {Object.keys(summary.type_performance || {}).length > 0 && (
              <div className="mi-overview-card mi-full-width">
                <h4><i className="fas fa-chart-bar" style={{ color: '#D4A853' }} /> Performance by Type</h4>
                <div className="mi-type-bars">
                  {Object.entries(summary.type_performance).map(([type, score]) => {
                    const color = score >= 80 ? '#22c55e' : score >= 60 ? '#f59e0b' : '#ef4444';
                    return (
                      <div key={type} className="mi-type-bar-row">
                        <span className="mi-type-bar-label">
                          <i className={TYPE_ICONS[type] || 'fas fa-question'} /> {type.replace(/-/g, ' ')}
                        </span>
                        <div className="mi-type-bar-wrap">
                          <div className="mi-type-bar-fill" style={{ width: `${score}%`, background: color }} />
                        </div>
                        <span className="mi-type-bar-score" style={{ color }}>{score}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'questions' && (
        <div className="mi-tab-content">
          <div className="mi-qa-list">
            {questions.map((q, i) => {
              // Support both integer and string keys
              const ev = evaluations[i] || evaluations[String(i)];
              const ans = answers[i] || answers[String(i)];
              const skipped = !ans?.trim();
              const grade = ev?.grade || (skipped ? 'F' : '—');
              const gcfg = GRADE_CONFIG[grade] || {};
              const isOpen = expandedQ === i;

              return (
                <div key={i} className={`mi-qa-item ${isOpen ? 'open' : ''} ${skipped ? 'skipped' : ''}`}>
                  <div className="mi-qa-header" onClick={() => setExpandedQ(isOpen ? null : i)}>
                    <div className="mi-qa-num" style={{ background: gcfg.bg || '#ffffff10', color: gcfg.color || '#888' }}>
                      {i + 1}
                    </div>
                    <div className="mi-qa-q">{q.question}</div>
                    <div className="mi-qa-badges">
                      <span className="mi-diff-badge sm" style={{ color: DIFF_CONFIG[q.difficulty]?.color || '#fff', background: `${DIFF_CONFIG[q.difficulty]?.color || '#888'}18` }}>
                        {q.difficulty}
                      </span>
                      {skipped ? (
                        <span className="mi-skipped-badge">Skipped</span>
                      ) : ev ? (
                        <span className="mi-grade-badge" style={{ background: gcfg.bg, color: gcfg.color }}>
                          {grade} · {ev.score}/100
                        </span>
                      ) : null}
                    </div>
                    <i className={`fas fa-chevron-${isOpen ? 'up' : 'down'} mi-qa-chevron`} />
                  </div>

                  {isOpen && (
                    <div className="mi-qa-body">
                      {/* User answer */}
                      <div className="mi-qa-answer-block">
                        <h6><i className="fas fa-user" /> Your Answer</h6>
                        {skipped ? (
                          <div className="mi-no-ans-block">
                            <i className="fas fa-ban" /> You skipped this question
                          </div>
                        ) : (
                          <p>{ans.trim()}</p>
                        )}
                      </div>

                      {/* Model answer — shown always */}
                      {ev?.ideal_answer && (
                        <div className="mi-qa-ideal-block">
                          <h6><i className="fas fa-star" style={{ color: '#D4A853' }} /> Model Answer</h6>
                          <p>{ev.ideal_answer}</p>
                        </div>
                      )}

                      {/* Feedback — only if answered */}
                      {!skipped && ev && (ev.strengths?.length > 0 || ev.improvements?.length > 0) && (
                        <div className="mi-qa-fb-row">
                          {ev.strengths?.length > 0 && (
                            <div className="mi-qa-fb strengths">
                              <h6><i className="fas fa-check" /> Strengths</h6>
                              <ul>{ev.strengths.map((s, j) => <li key={j}>{renderItem(s)}</li>)}</ul>
                            </div>
                          )}
                          {ev.improvements?.length > 0 && (
                            <div className="mi-qa-fb improvements">
                              <h6><i className="fas fa-arrow-up" /> What to Improve</h6>
                              <ul>{ev.improvements.map((s, j) => <li key={j}>{renderItem(s)}</li>)}</ul>
                            </div>
                          )}
                        </div>
                      )}

                      {/* For skipped: improvements */}
                      {skipped && ev?.improvements?.length > 0 && (
                        <div className="mi-qa-fb improvements full-width">
                          <h6><i className="fas fa-lightbulb" /> What to Study</h6>
                          <ul>{ev.improvements.map((s, j) => <li key={j}>{renderItem(s)}</li>)}</ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {activeTab === 'insights' && (
        <div className="mi-tab-content mi-insights">
          {loadingSummary ? (
            <div className="mi-loading-summary">
              <i className="fas fa-spinner fa-spin" /> Generating personalised AI insights...
            </div>
          ) : summary ? (
            <>
              {/* ── Stat Cards Row ── */}
              <div className="mi-insight-cards">
                <div className="mi-insight-card strongest">
                  <i className="fas fa-trophy" />
                  <h5>Strongest Area</h5>
                  <strong>{summary.strongest_area?.replace(/-/g, ' ') || '—'}</strong>
                </div>
                <div className="mi-insight-card weakest">
                  <i className="fas fa-bullseye" />
                  <h5>Focus Area</h5>
                  <strong>{summary.weakest_area?.replace(/-/g, ' ') || '—'}</strong>
                </div>
                <div className="mi-insight-card answered">
                  <i className="fas fa-check-circle" />
                  <h5>Completion</h5>
                  <strong>{summary.completion_rate}%</strong>
                </div>
                <div className="mi-insight-card score">
                  <i className="fas fa-star" />
                  <h5>Average Score</h5>
                  <strong>{summary.avg_score}%</strong>
                </div>
              </div>

              {/* ── Readiness Verdict ── */}
              {summary.readiness_verdict && (
                <div className="mi-readiness-verdict">
                  <div className="mi-readiness-icon">
                    <i className="fas fa-user-check" />
                  </div>
                  <div>
                    <h5>Interview Readiness</h5>
                    <p>{summary.readiness_verdict}</p>
                  </div>
                </div>
              )}

              {/* ── AI Narrative ── */}
              {summary.ai_narrative && (
                <div className="mi-insight-section narrative">
                  <h5><i className="fas fa-robot" /> AI Performance Analysis</h5>
                  <p>{summary.ai_narrative}</p>
                </div>
              )}

              {/* ── Skill Gaps + Study Plan side by side ── */}
              <div className="mi-insight-two-col">
                {summary.skill_gaps?.length > 0 && (
                  <div className="mi-insight-section gaps">
                    <h5><i className="fas fa-exclamation-triangle" /> Identified Skill Gaps</h5>
                    <ul className="mi-insight-list">
                      {summary.skill_gaps.map((g, i) => <li key={i}>{g}</li>)}
                    </ul>
                  </div>
                )}
                {summary.study_plan?.length > 0 && (
                  <div className="mi-insight-section study">
                    <h5><i className="fas fa-book-open" /> Personalised Study Plan</h5>
                    <ol className="mi-insight-list ordered">
                      {summary.study_plan.map((s, i) => <li key={i}>{s}</li>)}
                    </ol>
                  </div>
                )}
              </div>

              {/* ── Interview Tips ── */}
              {summary.interview_tips?.length > 0 && (
                <div className="mi-insight-section tips">
                  <h5><i className="fas fa-lightbulb" /> Interview Tips for You</h5>
                  <div className="mi-tips-grid">
                    {summary.interview_tips.map((tip, i) => (
                      <div key={i} className="mi-tip-card">
                        <span className="mi-tip-num">{i + 1}</span>
                        <p>{tip}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── Grade Distribution ── */}
              <div className="mi-grade-dist">
                <h5><i className="fas fa-chart-bar" /> Grade Distribution ({questions.length} questions)</h5>
                <div className="mi-grade-dist-bars">
                  {['A','B','C','D','F'].map(g => {
                    const count = summary.grade_counts?.[g] || 0;
                    const pct = questions.length > 0 ? (count / questions.length) * 100 : 0;
                    const gcfg = GRADE_CONFIG[g];
                    return (
                      <div key={g} className="mi-grade-dist-col">
                        <div className="mi-grade-dist-bar-wrap">
                          <div className="mi-grade-dist-bar" style={{ height: `${Math.max(pct, count > 0 ? 4 : 0)}%`, background: gcfg.color }} />
                        </div>
                        <span className="mi-grade-dist-label" style={{ color: gcfg.color }}>{g}</span>
                        <span className="mi-grade-dist-count">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          ) : (
            <p className="mi-empty">Could not load insights. Please try again.</p>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="mi-results-actions">
        <button className="mi-action-btn secondary" onClick={onNewResume}>
          <i className="fas fa-upload" /> Upload New Resume
        </button>
        <button className="mi-action-btn primary" onClick={onRetry}>
          <i className="fas fa-redo" /> Try Again
        </button>
      </div>
    </div>
  );
}

// ─── Root Component ───────────────────────────────────────────────────────────
const MockInterview = ({ user, onLogout, initialJobDescription = '', onBack }) => {
  const [stage, setStage] = useState('setup');   // setup | interview | results
  const [questions, setQuestions] = useState([]);
  const [jobDescription, setJobDescription] = useState(initialJobDescription);
  const [resultData, setResultData] = useState(null);

  // Redirect if no resume
  useEffect(() => {
    if (!user?.resume_filename) {
      alert('Please upload your resume first to start a mock interview.');
      if (onBack) onBack();
      else window.location.href = '/upload-resume';
    }
  }, [user, onBack]);

  const handleStart = (qs, jd) => {
    setQuestions(qs);
    setJobDescription(jd);
    setStage('interview');
  };

  const handleFinish = async (data) => {
    setResultData(data);
    setStage('results');

    try {
      // Calculate overall score from evaluations to save into history
      const evals = Object.values(data.evaluations || {});
      const finalScore = evals.length > 0 
        ? evals.reduce((sum, ev) => sum + (ev.score || 0), 0) / evals.length 
        : 0;

      await hrAPI.saveSession({
        session_type: 'mock',
        questions: data.questions,
        answers: {
          user_answers: data.answers,
          evaluations: data.evaluations,
          score: finalScore,
          duration: data.durationSeconds
        }
      });
    } catch (err) {
      console.warn('[MockInterview] Failed to save session to history:', err);
    }
  };

  const handleRetry = () => {
    setStage('setup');
    setQuestions([]);
    setResultData(null);
  };

  return (
    <div className="mi-root">
      {/* Optional back button */}
      {onBack && stage === 'setup' && (
        <button className="mi-back-btn" onClick={onBack}>
          <i className="fas fa-arrow-left" /> Back to Analysis
        </button>
      )}

      {stage === 'setup' && (
        <SetupStage initialJD={initialJobDescription} onStart={handleStart} />
      )}
      {stage === 'interview' && (
        <InterviewStage
          questions={questions}
          jobDescription={jobDescription}
          onFinish={handleFinish}
        />
      )}
      {stage === 'results' && (
        <ResultsStage
          data={resultData}
          onRetry={handleRetry}
          onNewResume={() => { if (onBack) onBack(); else window.location.href = '/upload-resume'; }}
        />
      )}
    </div>
  );
};

export default MockInterview;
