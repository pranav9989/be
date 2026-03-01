import React from 'react';
import './StatsCard.css';

const StatsCard = ({ number, label }) => {
  return (
    <div className="stat-card-component">
      <div className="stat-number">{number}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
};

export default StatsCard;