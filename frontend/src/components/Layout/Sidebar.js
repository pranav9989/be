import React from 'react';
import './Sidebar.css';

const Sidebar = ({ topics = [], onQuickQuestion }) => {
  const defaultTopics = [
    {
      name: 'DBMS',
      displayName: 'Database Management Systems',
      icon: 'fas fa-database',
      subtopics: ['Normal Forms', 'ER Modeling', 'Joins', 'ACID Properties']
    },
    {
      name: 'OOPs',
      displayName: 'Object-Oriented Programming',
      icon: 'fas fa-code',
      subtopics: ['Inheritance', 'Polymorphism', 'Encapsulation', 'Abstraction']
    },
    {
      name: 'OS',
      displayName: 'Operating Systems',
      icon: 'fas fa-cogs',
      subtopics: ['Process Management', 'Memory Management', 'Synchronization', 'File Systems']
    }
  ];

  const quickQuestions = [
    "What is normalization in DBMS?",
    "Explain polymorphism in OOP",
    "What is deadlock in operating systems?",
    "Difference between process and thread",
    "What are ACID properties in DBMS?",
    "What is virtual memory?"
  ];

  const displayTopics = topics.length > 0 ? topics : defaultTopics;

  return (
    <aside className="sidebar">
      <div className="sidebar-content">
        <h3><i className="fas fa-tags"></i> Topics</h3>
        <div className="topics-list">
          {displayTopics.map((topic, index) => (
            <div key={index} className="topic-card" data-topic={topic.name}>
              <div className="topic-header">
                <i className={topic.icon}></i>
                <span>{topic.displayName || topic.name}</span>
              </div>
              <div className="subtopics">
                {topic.subtopics.slice(0, 4).map((subtopic, subIndex) => (
                  <span key={subIndex} className="subtopic">{subtopic}</span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="quick-questions">
          <h4><i className="fas fa-lightbulb"></i> Quick Questions</h4>
          <div className="question-suggestions">
            {quickQuestions.map((question, index) => (
              <button 
                key={index}
                className="suggestion-btn"
                onClick={() => onQuickQuestion(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;