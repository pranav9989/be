import React, { useState, useEffect } from 'react';
import Header from '../Layout/Header';
import SkillsManager from './SkillsManager';
import { profileAPI, statsAPI } from '../../services/api';
import './Profile.css';

const Profile = ({ user, onLogout }) => {
  const [profile, setProfile] = useState(user);
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ sessions: 0 });
  const [message, setMessage] = useState({ type: '', text: '' });

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await statsAPI.getUserStats();
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const handleSave = async (formData) => {
    setLoading(true);
    try {
      await profileAPI.update(formData); // Removed unused 'response' variable
      setProfile(prev => ({ ...prev, ...formData }));
      setEditing(false);
      showMessage('success', 'Profile updated successfully!');
    } catch (error) {
      showMessage('error', 'Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 5000);
  };

  return (
    <div className="profile-container">
      <Header 
        user={user} 
        onLogout={onLogout} 
        title="Profile Management" 
        showBack={true}
      />

      <main className="profile-main">
        {message.text && (
          <div className={`message ${message.type}`}>
            {message.text}
          </div>
        )}

        <section className="profile-header">
          <div className="profile-avatar">
            <i className="fas fa-user"></i>
          </div>
          <h2>{profile.full_name || profile.username}</h2>
          <p>Member since {new Date(profile.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</p>
        </section>

        <div className="profile-content">
          {/* Personal Information Card - CHANGED TO profile-info-card */}
          <div className="profile-info-card">
            <div className="card-header">
              <h3><i className="fas fa-user"></i> Personal Information</h3>
              <button 
                onClick={() => setEditing(!editing)}
                className="edit-btn"
              >
                <i className="fas fa-edit"></i> {editing ? 'Cancel' : 'Edit'}
              </button>
            </div>

            {!editing ? (
              <div className="info-display">
                <div className="info-item">
                  <span className="info-label">Full Name</span>
                  <span className="info-value">{profile.full_name || 'Not provided'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Email</span>
                  <span className="info-value">{profile.email}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Phone</span>
                  <span className="info-value">{profile.phone || 'Not provided'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Experience</span>
                  <span className="info-value">{profile.experience_years || 0} years</span>
                </div>
              </div>
            ) : (
              <ProfileForm 
                profile={profile}
                onSave={handleSave}
                onCancel={() => setEditing(false)}
                loading={loading}
              />
            )}
          </div>

          {/* Skills Card */}
          <SkillsManager 
            skills={profile.skills || []}
            onSkillsUpdate={(newSkills) => {
              setProfile(prev => ({ ...prev, skills: newSkills }));
              handleSave({ skills: newSkills });
            }}
          />
        </div>

        {/* Statistics */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-number">{stats.sessions}</div>
            <div className="stat-label">Practice Sessions</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{profile.resume_filename ? '✓' : '✗'}</div>
            <div className="stat-label">Resume Uploaded</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">{profile.skills?.length || 0}</div>
            <div className="stat-label">Skills Listed</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">1</div>
            <div className="stat-label">Days Active</div>
          </div>
        </div>
      </main>
    </div>
  );
};

const ProfileForm = ({ profile, onSave, onCancel, loading }) => {
  const [formData, setFormData] = useState({
    full_name: profile.full_name || '',
    phone: profile.phone || '',
    experience_years: profile.experience_years || 0
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="profile-form">
      <div className="form-group">
        <label htmlFor="full_name">Full Name</label>
        <input
          type="text"
          id="full_name"
          value={formData.full_name}
          onChange={(e) => setFormData(prev => ({ ...prev, full_name: e.target.value }))}
          className="form-input"
        />
      </div>

      <div className="form-group">
        <label htmlFor="phone">Phone Number</label>
        <input
          type="tel"
          id="phone"
          value={formData.phone}
          onChange={(e) => setFormData(prev => ({ ...prev, phone: e.target.value }))}
          className="form-input"
        />
      </div>

      <div className="form-group">
        <label htmlFor="experience_years">Years of Experience</label>
        <input
          type="number"
          id="experience_years"
          value={formData.experience_years}
          onChange={(e) => setFormData(prev => ({ ...prev, experience_years: parseInt(e.target.value) || 0 }))}
          min="0"
          max="50"
          className="form-input"
        />
      </div>

      <div className="form-actions">
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Saving...' : 'Save Changes'}
        </button>
        <button type="button" onClick={onCancel} className="btn btn-secondary">
          Cancel
        </button>
      </div>
    </form>
  );
};

export default Profile;