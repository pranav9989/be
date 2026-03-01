import React, { useState } from 'react';
import { authAPI } from '../../services/api';
import LoadingSpinner from '../Layout/LoadingSpinner';
import './Auth.css';

const Login = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await authAPI.login(formData);
      
      if (response.data.success) {
        if (response.data.token) {
          localStorage.setItem('token', response.data.token);
        }
        onLogin(response.data.user);
      } else {
        setError(response.data.message || 'Login failed');
      }
    } catch (error) {
      setError(error.response?.data?.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingSpinner message="Signing in..." />;
  }

  return (
    <div className="auth-container">

      {/* ── LEFT PANEL: Neural Network / Brain Animation ── */}
      <div className="auth-panel auth-panel--login">
        <div className="auth-panel-content">
          <div className="panel-tagline">
            <span className="panel-tag">AI-Powered</span>
          </div>
          <h2 className="panel-title">Master Your<br />CS Interviews</h2>
          <p className="panel-subtitle">
            Practice with an adaptive AI interviewer that knows your weak spots
            and helps you improve with every session.
          </p>

          {/* Animated neural network SVG */}
          <div className="panel-illustration">
            <svg viewBox="0 0 420 380" xmlns="http://www.w3.org/2000/svg" className="neural-svg">
              {/* Animated connection lines */}
              <g className="nn-connections">
                {/* Layer 0→1 */}
                <line x1="60" y1="100" x2="180" y2="70"  className="nn-line" style={{animationDelay:'0s'}}/>
                <line x1="60" y1="100" x2="180" y2="140" className="nn-line" style={{animationDelay:'0.3s'}}/>
                <line x1="60" y1="190" x2="180" y2="70"  className="nn-line" style={{animationDelay:'0.6s'}}/>
                <line x1="60" y1="190" x2="180" y2="140" className="nn-line" style={{animationDelay:'0.9s'}}/>
                <line x1="60" y1="190" x2="180" y2="210" className="nn-line" style={{animationDelay:'0.2s'}}/>
                <line x1="60" y1="280" x2="180" y2="140" className="nn-line" style={{animationDelay:'0.5s'}}/>
                <line x1="60" y1="280" x2="180" y2="210" className="nn-line" style={{animationDelay:'0.8s'}}/>
                <line x1="60" y1="280" x2="180" y2="280" className="nn-line" style={{animationDelay:'1.1s'}}/>
                {/* Layer 1→2 */}
                <line x1="180" y1="70"  x2="300" y2="100" className="nn-line nn-line--mid" style={{animationDelay:'0.4s'}}/>
                <line x1="180" y1="70"  x2="300" y2="190" className="nn-line nn-line--mid" style={{animationDelay:'0.7s'}}/>
                <line x1="180" y1="140" x2="300" y2="100" className="nn-line nn-line--mid" style={{animationDelay:'1.0s'}}/>
                <line x1="180" y1="140" x2="300" y2="190" className="nn-line nn-line--mid" style={{animationDelay:'0.1s'}}/>
                <line x1="180" y1="210" x2="300" y2="190" className="nn-line nn-line--mid" style={{animationDelay:'0.4s'}}/>
                <line x1="180" y1="210" x2="300" y2="280" className="nn-line nn-line--mid" style={{animationDelay:'0.9s'}}/>
                <line x1="180" y1="280" x2="300" y2="190" className="nn-line nn-line--mid" style={{animationDelay:'0.6s'}}/>
                <line x1="180" y1="280" x2="300" y2="280" className="nn-line nn-line--mid" style={{animationDelay:'1.2s'}}/>
                {/* Layer 2→3 */}
                <line x1="300" y1="100" x2="370" y2="175" className="nn-line nn-line--out" style={{animationDelay:'0.2s'}}/>
                <line x1="300" y1="190" x2="370" y2="175" className="nn-line nn-line--out" style={{animationDelay:'0.5s'}}/>
                <line x1="300" y1="280" x2="370" y2="175" className="nn-line nn-line--out" style={{animationDelay:'0.8s'}}/>
              </g>

              {/* ── Nodes ── */}
              {/* Input layer */}
              <circle cx="60"  cy="100" r="14" className="nn-node nn-node--input" style={{animationDelay:'0s'}}/>
              <circle cx="60"  cy="190" r="14" className="nn-node nn-node--input" style={{animationDelay:'0.4s'}}/>
              <circle cx="60"  cy="280" r="14" className="nn-node nn-node--input" style={{animationDelay:'0.8s'}}/>
              {/* Hidden 1 */}
              <circle cx="180" cy="70"  r="16" className="nn-node nn-node--hidden" style={{animationDelay:'0.2s'}}/>
              <circle cx="180" cy="140" r="16" className="nn-node nn-node--hidden" style={{animationDelay:'0.5s'}}/>
              <circle cx="180" cy="210" r="16" className="nn-node nn-node--hidden" style={{animationDelay:'0.7s'}}/>
              <circle cx="180" cy="280" r="16" className="nn-node nn-node--hidden" style={{animationDelay:'1.0s'}}/>
              {/* Hidden 2 */}
              <circle cx="300" cy="100" r="16" className="nn-node nn-node--hidden2" style={{animationDelay:'0.3s'}}/>
              <circle cx="300" cy="190" r="16" className="nn-node nn-node--hidden2" style={{animationDelay:'0.6s'}}/>
              <circle cx="300" cy="280" r="16" className="nn-node nn-node--hidden2" style={{animationDelay:'0.9s'}}/>
              {/* Output */}
              <circle cx="370" cy="175" r="20" className="nn-node nn-node--output"/>

              {/* Travelling signal dots */}
              <circle r="4" className="nn-signal" style={{animationDelay:'0s'}}>
                <animateMotion dur="2.4s" repeatCount="indefinite" begin="0s">
                  <mpath href="#path-in1"/>
                </animateMotion>
              </circle>
              <circle r="4" className="nn-signal nn-signal--b" style={{animationDelay:'0.8s'}}>
                <animateMotion dur="2.8s" repeatCount="indefinite" begin="0.8s">
                  <mpath href="#path-mid1"/>
                </animateMotion>
              </circle>
              <circle r="4" className="nn-signal nn-signal--c" style={{animationDelay:'1.6s'}}>
                <animateMotion dur="2.2s" repeatCount="indefinite" begin="1.6s">
                  <mpath href="#path-out1"/>
                </animateMotion>
              </circle>

              {/* Signal paths (hidden) */}
              <defs>
                <path id="path-in1"  d="M60,190 L180,140"/>
                <path id="path-mid1" d="M180,140 L300,190"/>
                <path id="path-out1" d="M300,190 L370,175"/>
              </defs>

              {/* Input labels */}
              <text x="30" y="104" className="nn-label">DS</text>
              <text x="26" y="194" className="nn-label">OOP</text>
              <text x="30" y="284" className="nn-label">OS</text>
              {/* Output label */}
              <text x="358" y="205" className="nn-label nn-label--out">You</text>
            </svg>
          </div>

          {/* Feature pills */}
          <div className="panel-features">
            <span className="panel-feature"><i className="fas fa-check-circle"></i> Topic-adaptive questions</span>
            <span className="panel-feature"><i className="fas fa-check-circle"></i> Real semantic scoring</span>
            <span className="panel-feature"><i className="fas fa-check-circle"></i> Progress tracking</span>
          </div>
        </div>
      </div>

      {/* ── RIGHT PANEL: Form ── */}
      <div className="auth-card">
        <div className="auth-header">
          <i className="fas fa-brain"></i>
          <h1>Welcome Back</h1>
          <p>Sign in to continue to CS Interview Assistant</p>
        </div>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              required
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              required
              className="form-input"
            />
            <a href="/forgot-password" className="forgot-password-link">
              Forgot password?
            </a>
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Signing In...' : 'Sign In'}
          </button>
        </form>

        <div className="auth-links">
          <p>Don't have an account?</p>
          <a href="/signup">Create an account</a>
        </div>
      </div>
    </div>
  );
};

export default Login;