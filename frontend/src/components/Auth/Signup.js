import React, { useState } from 'react';
import { authAPI } from '../../services/api';
import LoadingSpinner from '../Layout/LoadingSpinner';
import './Auth.css';

const Signup = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    full_name: '',
    username: '',
    email: '',
    password: '',
    confirm_password: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [passwordStrength, setPasswordStrength] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });

    if (name === 'password') {
      setPasswordStrength(checkPasswordStrength(value));
    }
  };

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (!formData.full_name || !formData.username || !formData.email || !formData.password || !formData.confirm_password) {
      setError('Please fill in all fields');
      setLoading(false);
      return;
    }

    if (formData.password !== formData.confirm_password) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters long');
      setLoading(false);
      return;
    }

    if (!isValidEmail(formData.email)) {
      setError('Please enter a valid email address');
      setLoading(false);
      return;
    }

    try {
      const response = await authAPI.signup({
        full_name: formData.full_name,
        username: formData.username,
        email: formData.email,
        password: formData.password
      });
      
      if (response.data.success) {
        localStorage.setItem('token', response.data.token);
        onLogin(response.data.user);
      } else {
        setError(response.data.message || 'Registration failed');
      }
    } catch (error) {
      setError(error.response?.data?.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  const isValidEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const getPasswordStrengthText = () => {
    switch (passwordStrength) {
      case 'weak': return 'Weak password';
      case 'medium': return 'Medium password';
      case 'strong': return 'Strong password ✓';
      default: return '';
    }
  };

  if (loading) {
    return <LoadingSpinner message="Creating account..." />;
  }

  return (
    <div className="auth-container">

      {/* ── LEFT PANEL: Rocket / Journey Animation ── */}
      <div className="auth-panel auth-panel--signup">
        <div className="auth-panel-content">
          <div className="panel-tagline">
            <span className="panel-tag panel-tag--violet">Begin Today</span>
          </div>
          <h2 className="panel-title">Launch Your<br />Interview Journey</h2>
          <p className="panel-subtitle">
            Join thousands of students who've aced their CS interviews.
            Your personalised AI coach is ready.
          </p>

          {/* Animated rocket / journey SVG */}
          <div className="panel-illustration">
            <svg viewBox="0 0 420 380" xmlns="http://www.w3.org/2000/svg" className="rocket-svg">
              <defs>
                <radialGradient id="glow-a" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#D4A853" stopOpacity="0.4"/>
                  <stop offset="100%" stopColor="#D4A853" stopOpacity="0"/>
                </radialGradient>
                <radialGradient id="glow-b" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#7B6FEB" stopOpacity="0.35"/>
                  <stop offset="100%" stopColor="#7B6FEB" stopOpacity="0"/>
                </radialGradient>
                <filter id="blur-sm">
                  <feGaussianBlur stdDeviation="3"/>
                </filter>
              </defs>

              {/* Background stars */}
              <g className="stars">
                <circle cx="40"  cy="30"  r="2" className="star" style={{animationDelay:'0s'}}/>
                <circle cx="120" cy="15"  r="1.5" className="star" style={{animationDelay:'0.5s'}}/>
                <circle cx="200" cy="40"  r="2" className="star" style={{animationDelay:'1s'}}/>
                <circle cx="280" cy="20"  r="1" className="star" style={{animationDelay:'1.5s'}}/>
                <circle cx="360" cy="45"  r="1.5" className="star" style={{animationDelay:'0.3s'}}/>
                <circle cx="390" cy="10"  r="1" className="star" style={{animationDelay:'0.8s'}}/>
                <circle cx="70"  cy="80"  r="1.5" className="star" style={{animationDelay:'1.2s'}}/>
                <circle cx="380" cy="130" r="2" className="star" style={{animationDelay:'0.6s'}}/>
                <circle cx="30"  cy="160" r="1" className="star" style={{animationDelay:'1.7s'}}/>
                <circle cx="400" cy="220" r="1.5" className="star" style={{animationDelay:'0.2s'}}/>
                <circle cx="50"  cy="300" r="1" className="star" style={{animationDelay:'1.0s'}}/>
                <circle cx="100" cy="355" r="2" className="star" style={{animationDelay:'0.4s'}}/>
                <circle cx="350" cy="335" r="1.5" className="star" style={{animationDelay:'1.3s'}}/>
                <circle cx="320" cy="295" r="1" className="star" style={{animationDelay:'0.9s'}}/>
              </g>

              {/* Orbiting planet ring (decorative) */}
              <ellipse cx="210" cy="310" rx="95" ry="26"
                fill="none" stroke="url(#glow-b)" strokeWidth="2"
                className="orbit-ring"/>

              {/* Planet */}
              <circle cx="210" cy="310" r="38" className="planet"/>
              <ellipse cx="210" cy="310" rx="95" ry="9"
                fill="none" stroke="#7B6FEB" strokeWidth="1.5" opacity="0.5"
                className="planet-ring"/>
              {/* Planet surface detail */}
              <ellipse cx="196" cy="302" rx="14" ry="9" className="planet-crater"/>
              <ellipse cx="224" cy="320" rx="8"  ry="5" className="planet-crater" style={{opacity:0.5}}/>

              {/* Glow halos */}
              <circle cx="210" cy="145" r="70" fill="url(#glow-a)" className="rocket-glow" filter="url(#blur-sm)"/>

              {/* Rocket body */}
              <g className="rocket" transform="translate(210,145)">
                {/* Engine flame trail */}
                <g className="flame">
                  <ellipse cx="0" cy="65" rx="12" ry="22" className="flame-outer"/>
                  <ellipse cx="0" cy="62" rx="7"  ry="15" className="flame-inner"/>
                  <ellipse cx="0" cy="58" rx="3"  ry="8"  className="flame-core"/>
                </g>

                {/* Body */}
                <rect x="-18" y="-52" width="36" height="78" rx="4" className="rocket-body"/>
                {/* Nose cone */}
                <polygon points="0,-80 -18,-52 18,-52" className="rocket-nose"/>
                {/* Window */}
                <circle cx="0" cy="-22" r="10" className="rocket-window"/>
                <circle cx="0" cy="-22" r="6"  className="rocket-window-inner"/>
                {/* Fins */}
                <polygon points="-18,20 -38,50 -18,50" className="rocket-fin"/>
                <polygon points="18,20  38,50  18,50" className="rocket-fin"/>
                {/* Stripe */}
                <rect x="-18" y="-4" width="36" height="8" className="rocket-stripe"/>

                {/* Small particles */}
                <circle cx="-6" cy="85" r="2.5" className="exhaust-particle" style={{animationDelay:'0s'}}/>
                <circle cx="6"  cy="92" r="2"   className="exhaust-particle" style={{animationDelay:'0.3s'}}/>
                <circle cx="0"  cy="100" r="1.5" className="exhaust-particle" style={{animationDelay:'0.6s'}}/>
              </g>

              {/* Floating milestone tags */}
              <g className="milestone" style={{animationDelay:'0s'}}>
                <rect x="10" y="120" width="88" height="30" rx="8" className="milestone-bg"/>
                <text x="22" y="140" className="milestone-text">📚 Study Plan</text>
              </g>
              <g className="milestone" style={{animationDelay:'0.4s'}}>
                <rect x="322" y="165" width="88" height="30" rx="8" className="milestone-bg"/>
                <text x="333" y="185" className="milestone-text">🎯 Mock Tests</text>
              </g>
              <g className="milestone" style={{animationDelay:'0.8s'}}>
                <rect x="22" y="230" width="96" height="30" rx="8" className="milestone-bg"/>
                <text x="30" y="250" className="milestone-text">🏆 Offer Letter</text>
              </g>
            </svg>
          </div>

          {/* Stats */}
          <div className="panel-stats">
            <div className="panel-stat">
              <span className="panel-stat-number">3</span>
              <span className="panel-stat-label">Core Subjects</span>
            </div>
            <div className="panel-stat-divider"></div>
            <div className="panel-stat">
              <span className="panel-stat-number">∞</span>
              <span className="panel-stat-label">Practice Questions</span>
            </div>
            <div className="panel-stat-divider"></div>
            <div className="panel-stat">
              <span className="panel-stat-number">AI</span>
              <span className="panel-stat-label">Adaptive Coach</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── RIGHT PANEL: Form ── */}
      <div className="auth-card">
        <div className="auth-header">
          <i className="fas fa-user-plus"></i>
          <h1>Create Account</h1>
          <p>Join CS Interview Assistant today</p>
        </div>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="full_name">Full Name</label>
            <input
              type="text"
              id="full_name"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              required
              className="form-input"
            />
          </div>

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
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
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
            {passwordStrength && (
              <div className={`password-strength strength-${passwordStrength}`}>
                {getPasswordStrengthText()}
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
              required
              className="form-input"
            />
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <div className="auth-links">
          <p>Already have an account?</p>
          <a href="/login">Sign in here</a>
        </div>
      </div>
    </div>
  );
};

export default Signup;