import React from 'react';
import { useNavigate } from 'react-router-dom';
import './FeatureCard.css';

const FeatureCard = ({ title, description, icon, link, color }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(link);
  };

  return (
    <div className={`feature-card ${color}-card`} onClick={handleClick}>
      <i className={icon}></i>
      <h3>{title}</h3>
      <p>{description}</p>
      <button className="feature-btn">
        Get Started
      </button>
    </div>
  );
};

export default FeatureCard;