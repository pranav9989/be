import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { authAPI } from '../../services/api';
import './Auth.css';

const ResetPassword = () => {
  const [formData, setFormData] = useState({ new_password: '', confirm_password: '' });
  const [passwordStrength, setPasswordStrength] = useState('');
  const [token, setToken] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');
  const [countdown, setCountdown] = useState(5);

  const navigate = useNavigate();
  const location = useLocation();

  // Extract token from URL query string on mount
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const t = params.get('token');
    if (t) {
      setToken(t);
    } else {
      setError('No reset token found. Please use the link from your email.');
    }
  }, [location.search]);

  // Countdown redirect after success
  useEffect(() => {
    if (!success) return;
    if (countdown <= 0) { navigate('/login'); return; }
    const timer = setTimeout(() => setCountdown(c => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [success, countdown, navigate]);

  const checkPasswordStrength = (password) => {
    let score = 0;
    if (password.length >= 8) score++;
    if (password.match(/[a-z]/)) score++;
    if (password.match(/[A-Z]/)) score++;
    if (password.match(/[0-9]/)) score++;
    if (password.match(/[^a-zA-Z0-9]/)) score++;
    if (score <= 2) return 'weak';
    if (score <= 4) return 'medium';
    return 'strong';
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (name === 'new_password') setPasswordStrength(checkPasswordStrength(value));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (formData.new_password.length < 6) {
      setError('Password must be at least 6 characters'); return;
    }
    if (formData.new_password !== formData.confirm_password) {
      setError('Passwords do not match'); return;
    }

    setLoading(true);
    try {
      const response = await authAPI.resetPassword(token, formData.new_password);
      if (response.data.success) {
        setSuccess(true);
      } else {
        setError(response.data.message || 'Reset failed. Please try again.');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Reset failed. The link may have expired.');
    } finally {
      setLoading(false);
    }
  };

  const strengthText = { weak: 'Weak password', medium: 'Medium strength', strong: 'Strong password ✓' };

  return (
    <div className="auth-container">

      {/* ── LEFT PANEL – Shield / password strength theme ── */}
      <div className="auth-panel auth-panel--signup">
        <div className="auth-panel-content">
          <div className="panel-tagline">
            <span className="panel-tag panel-tag--violet">Almost there</span>
          </div>
          <h2 className="panel-title">Create a<br />Strong Password</h2>
          <p className="panel-subtitle">
            Choose a new password that's at least 8 characters and includes a
            mix of letters, numbers, and symbols for the best security.
          </p>

          <div className="panel-illustration">
            <svg viewBox="0 0 420 340" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <radialGradient id="shield-glow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#7B6FEB" stopOpacity="0.3"/>
                  <stop offset="100%" stopColor="#7B6FEB" stopOpacity="0"/>
                </radialGradient>
                <linearGradient id="shield-fill" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%"   stopColor="#2D2060"/>
                  <stop offset="100%" stopColor="#1A1035"/>
                </linearGradient>
              </defs>

              {/* Stars */}
              {[30,130,260,380,400,60,330,200].map((cx,i)=>(
                <circle key={i} cx={cx} cy={[40,20,35,50,130,220,270,290][i]}
                  r={[1.5,1,2,1,1.5,2,1,1.5][i]}
                  fill="rgba(242,242,245,0.45)"
                  style={{animation:'star-twinkle 2.5s ease-in-out infinite', animationDelay:`${i*0.35}s`}}/>
              ))}

              {/* Shield glow */}
              <circle cx="210" cy="155" r="110" fill="url(#shield-glow)"
                style={{animation:'glow-pulse 3s ease-in-out infinite'}}/>

              {/* Shield shape */}
              <path d="M210,55 L280,85 L280,155 Q280,215 210,245 Q140,215 140,155 L140,85 Z"
                fill="url(#shield-fill)" stroke="#7B6FEB" strokeWidth="2.5"
                style={{filter:'drop-shadow(0 0 14px rgba(123,111,235,0.5))', animation:'rocket-float 4s ease-in-out infinite'}}/>

              {/* Checkmark on shield */}
              <polyline points="182,155 205,178 242,130"
                fill="none" stroke="#9E95F5" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round"
                style={{animation:'nn-output-pulse 2s ease-in-out infinite'}}/>

              {/* Strength bars */}
              {['8+ Chars','Uppercase','Numbers','Symbols'].map((label, i) => (
                <g key={i}>
                  <rect x="60" y={270 + i * 0} width="300" height="0" rx="0" fill="none"/>
                  <rect x="60" y={272 + i * 0} width={[260,200,240,180][i]} height="6" rx="3"
                    fill={['#D4A853','#9E95F5','#4ADE80','#F472B6'][i]} opacity="0.7"
                    style={{animation:`nn-pulse 2.${i}s ease-in-out infinite`}}/>
                </g>
              ))}
              <text x="60" y="266" fill="rgba(242,242,245,0.5)"
                fontFamily="Inter,sans-serif" fontSize="10" fontWeight="600">
                PASSWORD STRENGTH FACTORS
              </text>
              {/* Bar labels */}
              {['8+ Chars','Uppercase','Numbers','Symbols'].map((label, i) => (
                <text key={i} x="370" y={[278,278,278,278][0] + 0}
                  fill="transparent" fontSize="0">{label}</text>
              ))}
            </svg>
          </div>

          <div className="panel-stats">
            <div className="panel-stat">
              <span className="panel-stat-number">6+</span>
              <span className="panel-stat-label">Min Characters</span>
            </div>
            <div className="panel-stat-divider"></div>
            <div className="panel-stat">
              <span className="panel-stat-number">AES</span>
              <span className="panel-stat-label">Encrypted Storage</span>
            </div>
            <div className="panel-stat-divider"></div>
            <div className="panel-stat">
              <span className="panel-stat-number">1×</span>
              <span className="panel-stat-label">One-time Link</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── RIGHT PANEL ── */}
      <div className="auth-card">
        {!success ? (
          <>
            <div className="auth-header">
              <i className="fas fa-lock-open"></i>
              <h1>Reset Password</h1>
              <p>Enter and confirm your new password below</p>
            </div>

            {error && <div className="error-message">{error}</div>}

            {!token && !error ? (
              <p style={{textAlign:'center', color:'var(--text-muted)', fontSize:'0.9rem'}}>
                Looking for your reset token…
              </p>
            ) : (
              <form onSubmit={handleSubmit} className="auth-form">
                <div className="form-group">
                  <label htmlFor="new_password">New Password</label>
                  <input
                    type="password"
                    id="new_password"
                    name="new_password"
                    value={formData.new_password}
                    onChange={handleChange}
                    placeholder="Min 6 characters"
                    required
                    className="form-input"
                    autoFocus
                  />
                  {passwordStrength && (
                    <div className={`password-strength strength-${passwordStrength}`}>
                      {strengthText[passwordStrength]}
                    </div>
                  )}
                </div>

                <div className="form-group">
                  <label htmlFor="confirm_password">Confirm Password</label>
                  <input
                    type="password"
                    id="confirm_password"
                    name="confirm_password"
                    value={formData.confirm_password}
                    onChange={handleChange}
                    placeholder="Repeat your new password"
                    required
                    className="form-input"
                  />
                </div>

                <button type="submit" className="btn btn-primary" disabled={loading || !token}>
                  {loading ? (
                    <><i className="fas fa-circle-notch fa-spin"></i> Updating…</>
                  ) : (
                    <><i className="fas fa-check-circle"></i> Set New Password</>
                  )}
                </button>
              </form>
            )}
          </>
        ) : (
          /* ── Success state ── */
          <div className="fp-success">
            <div className="fp-success-icon fp-success-icon--green">
              <i className="fas fa-check-circle"></i>
            </div>
            <h2>Password Updated!</h2>
            <p>
              Your password has been successfully reset.
              Redirecting you to sign in in{' '}
              <strong style={{color: 'var(--accent-gold)'}}>{countdown}s</strong>…
            </p>
            <button className="btn btn-primary" onClick={() => navigate('/login')}>
              <i className="fas fa-sign-in-alt"></i> Go to Sign In
            </button>
          </div>
        )}

        {!success && (
          <div className="auth-links">
            <p>Remembered it?</p>
            <a href="/login">Back to Sign In</a>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResetPassword;
