import React from 'react';
import { useNavigate } from 'react-router-dom';
import './FeatureCard.css';

const FeatureCard = ({ title, description, icon, link, color }) => {
  const navigate = useNavigate();

  const handleButtonClick = (e) => {
    e.stopPropagation();
    navigate(link);
  };

  return (
    <div className={`feature-card color-${color}`} onClick={() => navigate(link)}>
      <div className="feature-icon">
        <i className={icon}></i>
      </div>
      <div className="feature-title">{title}</div>
      <p className="feature-description">{description}</p>
      <span className="feature-arrow">
        <i className="fas fa-arrow-right"></i>
      </span>
    </div>
  );
};

export default FeatureCard;