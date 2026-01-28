import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { authAPI } from './services/api';
import Login from './components/Auth/Login';
import Signup from './components/Auth/Signup';
import Dashboard from './components/Dashboard/Dashboard';
import ChatInterface from './components/Chat/ChatInterface';
import Profile from './components/Profile/Profile';
import HRInterview from './components/HRInterview/HRInterview';
import ResumeUpload from './components/Resume/ResumeUpload';
import ConversationalInterview from './components/ConversationalInterview/ConversationalInterview';
import InterviewStreamer from './components/LiveStreamingInterview/InterviewStreamer';
import LoadingSpinner from './components/Layout/LoadingSpinner';
import AgenticInterview from './components/AgenticInterview/AgenticInterview';
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      if (token) {
        const response = await authAPI.getProfile();
        setUser(response.data.user);
      }
    } catch (error) {
      localStorage.removeItem('token');
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setUser(null);
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route
            path="/login"
            element={!user ? <Login onLogin={handleLogin} /> : <Navigate to="/dashboard" />}
          />
          <Route
            path="/signup"
            element={!user ? <Signup onLogin={handleLogin} /> : <Navigate to="/dashboard" />}
          />
          <Route
            path="/dashboard"
            element={user ? <Dashboard user={user} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
          <Route
            path="/technical-chat"
            element={user ? <ChatInterface user={user} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
          <Route
            path="/profile"
            element={user ? <Profile user={user} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
          <Route
            path="/hr-interview"
            element={user ? <HRInterview user={user} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
          <Route
            path="/upload-resume"
            element={user ? <ResumeUpload user={user} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
          <Route
            path="/conversational-interview"
            element={user ? <ConversationalInterview user={user} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
          <Route
            path="/live-streaming-interview"
            element={user ? <InterviewStreamer userId={user.id} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
          <Route path="/" element={<Navigate to={user ? "/dashboard" : "/login"} />} />
          <Route
            path="/agentic-interview"
            element={user ? <AgenticInterview user={user} onLogout={handleLogout} /> : <Navigate to="/login" />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;