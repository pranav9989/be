import React from 'react';
import { useNavigate } from 'react-router-dom';
import './FeatureCard.css';

const FeatureCard = ({ title, description, icon, link, color }) => {
  const navigate = useNavigate();

  const handleCardClick = () => {
    navigate(link);
  };

  const handleButtonClick = (e) => {
    e.stopPropagation(); // Prevent card click when button is clicked
    navigate(link);
  };

  return (
    <div className={`feature-card ${color}-card`} onClick={handleCardClick}>
      <i className={icon}></i>
      <h3>{title}</h3>
      <p>{description}</p>
      <button className="feature-btn" onClick={handleButtonClick}>
        Get Started
      </button>
    </div>
  );
};

export default FeatureCard;