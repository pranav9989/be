import React, { useState, useEffect } from 'react';
import './SkillsManager.css';

const SkillsManager = ({ skills = [], onSkillsUpdate }) => {
  const [editing, setEditing] = useState(false);
  const [newSkill, setNewSkill] = useState('');
  const [currentSkills, setCurrentSkills] = useState(skills);
  const [selectedCategory, setSelectedCategory] = useState('programming');
  const [showDropdown, setShowDropdown] = useState(false);

  // Comprehensive technical skills database
  const technicalSkills = {
    programming: [
      'Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'C#', 'Go', 'Rust',
      'PHP', 'Ruby', 'Swift', 'Kotlin', 'R', 'MATLAB', 'Scala', 'Perl'
    ],
    web: [
      'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django', 'Flask',
      'Spring Boot', 'ASP.NET', 'Laravel', 'Ruby on Rails', 'Next.js', 'Nuxt.js',
      'HTML5', 'CSS3', 'SASS/SCSS', 'Tailwind CSS', 'Bootstrap', 'Material-UI'
    ],
    database: [
      'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle', 'SQL Server',
      'Elasticsearch', 'Cassandra', 'DynamoDB', 'Firebase', 'Supabase'
    ],
    cloud: [
      'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform',
      'Jenkins', 'GitLab CI', 'GitHub Actions', 'Ansible', 'Helm'
    ],
    data: [
      'Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch', 'Keras',
      'Jupyter', 'Tableau', 'Power BI', 'Apache Spark', 'Hadoop', 'Kafka'
    ],
    tools: [
      'Git', 'GitHub', 'Bitbucket', 'Jira', 'Confluence', 'Slack', 'VS Code',
      'IntelliJ', 'Postman', 'Figma', 'Adobe XD', 'Photoshop', 'Linux', 'Bash'
    ],
    soft: [
      'Agile', 'Scrum', 'Kanban', 'Project Management', 'Team Leadership',
      'Communication', 'Problem Solving', 'Critical Thinking', 'Time Management'
    ]
  };

  // Sync skills when prop changes
  useEffect(() => {
    setCurrentSkills(skills);
  }, [skills]);

  const handleAddSkill = (skill = null) => {
    const skillToAdd = skill || newSkill.trim();
    if (skillToAdd && !currentSkills.includes(skillToAdd)) {
      const updatedSkills = [...currentSkills, skillToAdd];
      setCurrentSkills(updatedSkills);
      setNewSkill('');
      setShowDropdown(false);
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

  const handleSkillSelect = (skill) => {
    handleAddSkill(skill);
  };

  const toggleDropdown = () => {
    setShowDropdown(!showDropdown);
  };

  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    setShowDropdown(true);
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
          {/* Category Selector */}
          <div className="skills-category-selector">
            <label className="category-label">Select Skill Category:</label>
            <div className="category-buttons">
              {Object.keys(technicalSkills).map((category) => (
                <button
                  key={category}
                  onClick={() => handleCategoryChange(category)}
                  className={`category-btn ${selectedCategory === category ? 'active' : ''}`}
                  type="button"
                >
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Skills Dropdown */}
          <div className="skills-dropdown-container">
            <div className="dropdown-header">
              <span className="dropdown-label">Choose from {selectedCategory} skills:</span>
              <button
                onClick={toggleDropdown}
                className="dropdown-toggle"
                type="button"
              >
                <i className={`fas fa-chevron-${showDropdown ? 'up' : 'down'}`}></i>
              </button>
            </div>

            {showDropdown && (
              <div className="skills-dropdown">
                {technicalSkills[selectedCategory].map((skill) => (
                  <button
                    key={skill}
                    onClick={() => handleSkillSelect(skill)}
                    className="dropdown-skill-item"
                    disabled={currentSkills.includes(skill)}
                    type="button"
                  >
                    {skill}
                    {currentSkills.includes(skill) && <i className="fas fa-check"></i>}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Custom Skill Input */}
          <div className="custom-skill-input">
            <label className="input-label">Or add a custom skill:</label>
            <div className="skills-input-group">
              <input
                type="text"
                value={newSkill}
                onChange={(e) => setNewSkill(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Enter custom skill (e.g., Advanced Excel)"
                className="form-input"
              />
              <button
                onClick={() => handleAddSkill()}
                className="add-skill-btn"
                disabled={!newSkill.trim()}
                type="button"
              >
                <i className="fas fa-plus"></i> Add
              </button>
            </div>
          </div>

          {/* Current Skills Display */}
          <div className="current-skills-section">
            <h4 className="section-title">
              <i className="fas fa-tags"></i> Your Skills ({currentSkills.length})
            </h4>
            <div className="skills-list-edit">
              {currentSkills.length > 0 ? (
                currentSkills.map((skill, index) => (
                  <span key={index} className="skill-tag-editable">
                    {skill}
                    <button
                      onClick={() => handleRemoveSkill(index)}
                      className="skill-remove"
                      title="Remove skill"
                      type="button"
                    >
                      <i className="fas fa-times"></i>
                    </button>
                  </span>
                ))
              ) : (
                <div className="no-skills-message">
                  <i className="fas fa-info-circle"></i>
                  No skills added yet. Choose from the categories above or add custom skills.
                </div>
              )}
            </div>
          </div>

          <div className="form-actions">
            <button
              onClick={() => setEditing(false)}
              className="btn btn-primary"
              type="button"
            >
              <i className="fas fa-check"></i> Done Editing
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SkillsManager;