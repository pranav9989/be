import React, { useState, useEffect } from 'react';
import Header from '../Layout/Header';
import StatsCard from './StatsCard';
import FeatureCard from './FeatureCard';
import { statsAPI } from '../../services/api';
import './Dashboard.css';

const Dashboard = ({ user, onLogout }) => {
  const [stats, setStats] = useState({
    sessions: 0,
    questions: 0,
    avg_score: null
  });

  useEffect(() => {
    loadUserStats();
  }, []);

  const loadUserStats = async () => {
    try {
      const response = await statsAPI.getUserStats();
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const features = [
    {
      title: "Technical Interview",
      description: "Practice with our AI-powered technical questions on DBMS, OOPs, and Operating Systems.",
      icon: "fas fa-code",
      link: "/technical-chat",
      color: "technical"
    },
    {
      title: "HR Interview",
      description: "Prepare for behavioral and HR questions tailored to your profile and experience.",
      icon: "fas fa-users",
      link: "/hr-interview",
      color: "hr"
    },
    {
      title: "Resume Analysis & Mock Interview",
      description: "Upload your resume for intelligent parsing and start personalized mock interviews based on your profile.",
      icon: "fas fa-file-upload",
      link: "/upload-resume",
      color: "resume"
    },
    {
      title: "Profile Management",
      description: "Update your skills, experience, and preferences to get better recommendations.",
      icon: "fas fa-user-cog",
      link: "/profile",
      color: "profile"
    },
    {
      title: "Conversational AI Interviewer",
      description: "Engage in real-time spoken interviews with AI, get instant feedback and analysis.",
      icon: "fas fa-microphone-alt",
      link: "/conversational-interview",
      color: "hr" // Using 'hr' color for now, can be changed later
    }
  ];

  return (
    <div className="dashboard-container">
      <Header 
        user={user} 
        onLogout={onLogout} 
        title="CS Interview Assistant" 
      />

      <main className="dashboard-main">
        <section className="welcome-section">
          <h2>Welcome back, {user.full_name || user.username}!</h2>
          <p>Ready to ace your next interview? Choose from our comprehensive preparation tools.</p>
        </section>

        <section className="features-grid">
          {features.map((feature, index) => (
            <FeatureCard key={index} {...feature} />
          ))}
        </section>

        <section className="stats-section">
          <div className="stats-header">
            <h3>Your Progress</h3>
            <p>Track your interview preparation journey</p>
          </div>
          <div className="stats-grid">
            <StatsCard 
              number={stats.sessions} 
              label="Practice Sessions" 
            />
            <StatsCard 
              number={stats.questions} 
              label="Questions Answered" 
            />
            <StatsCard 
              number={stats.avg_score ? `${stats.avg_score.toFixed(1)}%` : '-'} 
              label="Average Score" 
            />
            <StatsCard 
              number={1} 
              label="Days Active" 
            />
          </div>
        </section>
      </main>
    </div>
  );
};

export default Dashboard;