import React, { useState } from 'react';
import { authAPI } from '../../services/api';
import './Auth.css';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim()) { setError('Please enter your email address'); return; }
    setLoading(true);
    setError('');
    try {
      await authAPI.forgotPassword(email.trim());
      setSubmitted(true);
    } catch (err) {
      setError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">

      {/* ── LEFT PANEL (same neural-network as Login) ── */}
      <div className="auth-panel auth-panel--login">
        <div className="auth-panel-content">
          <div className="panel-tagline">
            <span className="panel-tag">Account Recovery</span>
          </div>
          <h2 className="panel-title">Don't Worry,<br />We've Got You</h2>
          <p className="panel-subtitle">
            It happens to everyone. Enter your email and we'll send you a secure
            link to reset your password in seconds.
          </p>

          <div className="panel-illustration">
            <svg viewBox="0 0 420 320" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <radialGradient id="lock-glow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#D4A853" stopOpacity="0.25"/>
                  <stop offset="100%" stopColor="#D4A853" stopOpacity="0"/>
                </radialGradient>
              </defs>

              {/* Background dots */}
              {[40,100,180,260,360,390,20,310,140,80].map((cx, i) => (
                <circle key={i} cx={cx} cy={[60,30,50,25,55,120,200,200,270,300][i]}
                  r={[2,1.5,2,1,1.5,2,1,1.5,2,1][i]}
                  fill="rgba(242,242,245,0.4)"
                  style={{animation:`star-twinkle 2.5s ease-in-out infinite`, animationDelay:`${i*0.3}s`}}/>
              ))}

              {/* Glow halo */}
              <circle cx="210" cy="160" r="100" fill="url(#lock-glow)"
                style={{animation:'glow-pulse 3s ease-in-out infinite'}}/>

              {/* Lock body */}
              <rect x="155" y="170" width="110" height="90" rx="16"
                fill="#1A1035" stroke="#D4A853" strokeWidth="2"
                style={{filter:'drop-shadow(0 0 12px rgba(212,168,83,0.4))'}}/>

              {/* Lock shackle (arc) */}
              <path d="M 175 170 L 175 140 A 35 35 0 0 1 245 140 L 245 170"
                fill="none" stroke="#D4A853" strokeWidth="5" strokeLinecap="round"
                style={{animation:'nn-output-pulse 2s ease-in-out infinite'}}/>

              {/* Keyhole */}
              <circle cx="210" cy="210" r="14" fill="#D4A853" opacity="0.9"/>
              <rect x="205" y="220" width="10" height="18" rx="3" fill="#D4A853" opacity="0.9"/>

              {/* Orbiting email icon */}
              <g style={{animation:'rocket-float 4s ease-in-out infinite', transformOrigin:'210px 160px'}}>
                <rect x="292" y="110" width="56" height="40" rx="8"
                  fill="rgba(123,111,235,0.2)" stroke="#7B6FEB" strokeWidth="1.5"/>
                <polyline points="292,110 320,132 348,110"
                  fill="none" stroke="#9E95F5" strokeWidth="1.5"/>
              </g>

              {/* Three secure checkmarks below */}
              {['Secure Token', '15-min Expiry', 'One-time Use'].map((label, i) => (
                <g key={i} style={{animation:`milestone-float 3s ease-in-out infinite`, animationDelay:`${i*0.5}s`}}>
                  <rect x={50 + i * 120} y="286" width="105" height="24" rx="6"
                    fill="rgba(15,10,40,0.7)" stroke="rgba(212,168,83,0.3)" strokeWidth="1"/>
                  <text x={80 + i * 120} y="302" fill="rgba(242,242,245,0.75)"
                    fontFamily="Inter,sans-serif" fontSize="10" fontWeight="600">✓ {label}</text>
                </g>
              ))}
            </svg>
          </div>

          <div className="panel-features">
            <span className="panel-feature"><i className="fas fa-shield-alt"></i> Cryptographically secure token</span>
            <span className="panel-feature"><i className="fas fa-clock"></i> Link expires in 15 minutes</span>
            <span className="panel-feature"><i className="fas fa-ban"></i> One-time use — can't be reused</span>
          </div>
        </div>
      </div>

      {/* ── RIGHT PANEL ── */}
      <div className="auth-card">
        {!submitted ? (
          <>
            <div className="auth-header">
              <i className="fas fa-key"></i>
              <h1>Forgot Password?</h1>
              <p>Enter your account email and we'll send a reset link</p>
            </div>

            {error && <div className="error-message">{error}</div>}

            <form onSubmit={handleSubmit} className="auth-form">
              <div className="form-group">
                <label htmlFor="email">Email Address</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  className="form-input"
                  autoFocus
                />
              </div>

              <button type="submit" className="btn btn-primary" disabled={loading}>
                {loading ? (
                  <><i className="fas fa-circle-notch fa-spin"></i> Sending…</>
                ) : (
                  <><i className="fas fa-paper-plane"></i> Send Reset Link</>
                )}
              </button>
            </form>
          </>
        ) : (
          /* ── Success state ── */
          <div className="fp-success">
            <div className="fp-success-icon">
              <i className="fas fa-envelope-open-text"></i>
            </div>
            <h2>Check Your Inbox</h2>
            <p>
              If <strong>{email}</strong> is registered, you'll receive a password
              reset link within a few seconds.
            </p>
            <ul className="fp-tips">
              <li><i className="fas fa-folder"></i> Check your spam / junk folder</li>
              <li><i className="fas fa-clock"></i> Link is valid for 15 minutes</li>
            </ul>
            <button className="btn btn-secondary"
              onClick={() => { setSubmitted(false); setEmail(''); }}>
              Try a different email
            </button>
          </div>
        )}

        <div className="auth-links" style={{marginTop: submitted ? '1.5rem' : undefined}}>
          <p>Remember your password?</p>
          <a href="/login">Back to Sign In</a>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
