import React from 'react';
import './QuestionCard.css';

const QuestionCard = ({ question, index, answer, onAnswerChange }) => {
  const handleAnswerChange = (e) => {
    onAnswerChange(index, e.target.value);
  };

  return (
    <div className="question-card">
      <div className="question-header">
        <span className="question-number">Question {index + 1}</span>
        <span className="question-type">{question.type || 'General'}</span>
      </div>
      <div className="question-text">{question.question}</div>
      <div className="answer-section">
        <textarea
          className="answer-textarea"
          value={answer}
          onChange={handleAnswerChange}
          placeholder="Type your answer here..."
          rows={6}
        />
      </div>
    </div>
  );
};

export default QuestionCard;