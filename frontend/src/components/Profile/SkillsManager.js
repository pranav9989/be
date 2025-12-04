import React, { useState } from 'react';
import './SkillsManager.css';

const SkillsManager = ({ skills = [], onSkillsUpdate }) => {
  const [editing, setEditing] = useState(false);
  const [newSkill, setNewSkill] = useState('');
  const [currentSkills, setCurrentSkills] = useState(skills);

  const handleAddSkill = () => {
    if (newSkill.trim() && !currentSkills.includes(newSkill.trim())) {
      const updatedSkills = [...currentSkills, newSkill.trim()];
      setCurrentSkills(updatedSkills);
      setNewSkill('');
      onSkillsUpdate(updatedSkills);
    }
  };

  const handleRemoveSkill = (index) => {
    const updatedSkills = currentSkills.filter((_, i) => i !== index);
    setCurrentSkills(updatedSkills);
    onSkillsUpdate(updatedSkills);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddSkill();
    }
  };

  return (
    <div className="skills-manager">
      <div className="card-header">
        <h3><i className="fas fa-cogs"></i> Skills & Expertise</h3>
        <button 
          onClick={() => setEditing(!editing)}
          className="edit-btn"
        >
          <i className="fas fa-edit"></i> {editing ? 'Cancel' : 'Edit'}
        </button>
      </div>

      {!editing ? (
        <div className="skills-display">
          <div className="skills-list">
            {currentSkills.length > 0 ? (
              currentSkills.map((skill, index) => (
                <span key={index} className="skill-tag">
                  {skill}
                </span>
              ))
            ) : (
              <p className="no-skills">No skills added yet. Upload your resume or add them manually.</p>
            )}
          </div>
        </div>
      ) : (
        <div className="skills-edit">
          <div className="skills-input-group">
            <input
              type="text"
              value={newSkill}
              onChange={(e) => setNewSkill(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter a skill (e.g., Python, React)"
              className="form-input"
            />
            <button onClick={handleAddSkill} className="add-skill-btn">
              <i className="fas fa-plus"></i> Add
            </button>
          </div>
          
          <div className="skills-list-edit">
            {currentSkills.map((skill, index) => (
              <span key={index} className="skill-tag-editable">
                {skill}
                <button 
                  onClick={() => handleRemoveSkill(index)}
                  className="skill-remove"
                >
                  <i className="fas fa-times"></i>
                </button>
              </span>
            ))}
          </div>

          <div className="form-actions">
            <button 
              onClick={() => setEditing(false)}
              className="btn btn-primary"
            >
              Done Editing
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SkillsManager;