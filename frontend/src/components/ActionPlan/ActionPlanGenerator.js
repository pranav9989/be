import React, { useState } from 'react';
import api from '../../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './ActionPlanGenerator.css';

const ActionPlanGenerator = () => {
    const [days, setDays] = useState(7);
    const [selectedTopics, setSelectedTopics] = useState(['General Programming']);
    const [plan, setPlan] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    
    // History state
    const [savedPlans, setSavedPlans] = useState([]);
    const [loadingHistory, setLoadingHistory] = useState(false);
    const [expandedPlanId, setExpandedPlanId] = useState(null);

    const togglePlanExpansion = (id) => {
        setExpandedPlanId(expandedPlanId === id ? null : id);
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return '';
        try {
            const str = dateStr.endsWith('Z') || dateStr.includes('+') ? dateStr : dateStr + 'Z';
            const d = new Date(str);
            if (isNaN(d.getTime())) return dateStr.split('T')[0];
            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        } catch {
            return dateStr.split('T')[0];
        }
    };

    // Helper to remove AI code fences that break markdown rendering
    const scrubMarkdown = (text) => {
        if (!text) return '';
        // Remove ```markdown or ``` plus any trailing ```
        let scrubbed = text.trim();
        if (scrubbed.startsWith('```markdown')) {
            scrubbed = scrubbed.replace(/^```markdown\n?/, '').replace(/\n?```$/, '');
        } else if (scrubbed.startsWith('```')) {
            scrubbed = scrubbed.replace(/^```\n?/, '').replace(/\n?```$/, '');
        }
        return scrubbed;
    };

    // Fetch history
    const fetchHistory = async () => {
        setLoadingHistory(true);
        try {
            const response = await api.get('/profile/action_plans');
            if (response.data.success) {
                setSavedPlans(response.data.action_plans);
            }
        } catch (err) {
            console.error("Failed to fetch action plan history:", err);
        } finally {
            setLoadingHistory(false);
        }
    };

    React.useEffect(() => {
        fetchHistory();
    }, []);

    const availableTopics = [
        'General Programming',
        'Database Management Systems',
        'Operating Systems',
        'Object-Oriented Programming',
        'Computer Networks'
    ];

    const toggleTopic = (topic) => {
        setSelectedTopics(prev => 
            prev.includes(topic) 
                ? prev.filter(t => t !== topic)
                : [...prev, topic]
        );
    };

    const generatePlan = async () => {
        if (selectedTopics.length === 0) {
            setError('Please select at least one topic.');
            return;
        }

        setLoading(true);
        setError('');
        setPlan('');

        try {
            const response = await api.post(
                '/generate_action_plan',
                { days, topics: selectedTopics }
            );

            if (response.data.success) {
                setPlan(response.data.plan_markdown);
                fetchHistory(); // Refresh history list
            } else {
                setError(response.data.error || 'Failed to generate plan.');
            }
        } catch (err) {
            console.error(err);
            setError('Error connecting to the server.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="action-plan-container">
            <div className="action-plan-header">
                <h2>Personalized Action Plan</h2>
                <p>Generate an adaptive study schedule targeting your weakest concepts.</p>
            </div>

            <div className="action-plan-layout">
                {/* Controls */}
                <div className="action-plan-controls">
                    <div className="control-group">
                        <label className="control-label">Study Duration (Days)</label>
                        <input 
                            type="number" 
                            min="1" 
                            max="30"
                            value={days}
                            onChange={(e) => setDays(parseInt(e.target.value) || 7)}
                            className="duration-input"
                        />
                    </div>

                    <div className="control-group">
                        <label className="control-label">Focus Topics</label>
                        <div className="topics-list">
                            {availableTopics.map(topic => (
                                <label key={topic} className="topic-checkbox-label">
                                    <input 
                                        type="checkbox"
                                        checked={selectedTopics.includes(topic)}
                                        onChange={() => toggleTopic(topic)}
                                        className="topic-checkbox"
                                    />
                                    <span>{topic}</span>
                                </label>
                            ))}
                        </div>
                    </div>

                    <button 
                        onClick={generatePlan}
                        disabled={loading}
                        className="btn-generate"
                    >
                        {loading ? (
                            <>
                                <i className="fas fa-circle-notch fa-spin"></i>
                                Generating Plan...
                            </>
                        ) : (
                            <>
                                <i className="fas fa-bolt"></i>
                                Generate Plan
                            </>
                        )}
                    </button>

                    {error && (
                        <div className="error-message">
                            <i className="fas fa-exclamation-circle"></i>
                            <span>{error}</span>
                        </div>
                    )}
                </div>

                {/* Result View */}
                <div className="action-plan-result">
                    {loading ? (
                        <div className="loading-state">
                            <div className="action-plan-spinner"></div>
                            <h3>Analyzing your performance data...</h3>
                            <p>We're reviewing your past interview responses to identify weaknesses and build a custom study schedule.</p>
                        </div>
                    ) : plan ? (
                        <div className="result-markdown">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{scrubMarkdown(plan)}</ReactMarkdown>
                        </div>
                    ) : (
                        <div className="empty-state">
                            <i className="fas fa-clipboard-list" style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}></i>
                            <p>Select your topics and duration on the left to generate your personalized action plan.</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Saved Plans History Map */}
            <div className="action-plan-history">
                <h3>Your Saved Action Plans</h3>
                {loadingHistory ? (
                    <div className="loading-state"><div className="action-plan-spinner" style={{ width: '24px', height: '24px' }}></div></div>
                ) : savedPlans.length > 0 ? (
                    <div className="history-list">
                        {savedPlans.map(p => (
                            <div key={p.id} className={`history-item ${expandedPlanId === p.id ? 'expanded' : ''}`}>
                                <div 
                                    className="history-item-header" 
                                    onClick={() => togglePlanExpansion(p.id)}
                                    style={{ cursor: 'pointer' }}
                                >
                                    <div className="history-header-left">
                                        <div className="history-title">
                                            <i className={`fas fa-chevron-${expandedPlanId === p.id ? 'down' : 'right'}`} style={{ marginRight: '0.5rem', fontSize: '0.8rem', opacity: 0.7 }}></i>
                                            {p.days}-Day Plan
                                        </div>
                                        <div className="history-date">
                                            {formatDate(p.created_at)}
                                        </div>
                                    </div>
                                    <div className="history-topics">
                                        {p.topics.slice(0, 2).map((t, i) => (
                                            <span key={i} className="topic-tag">{t}</span>
                                        ))}
                                        {p.topics.length > 2 && <span className="topic-tag">+{p.topics.length - 2} more</span>}
                                    </div>
                                </div>
                                
                                {expandedPlanId === p.id && (
                                    <div className="history-preview">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{scrubMarkdown(p.plan_markdown)}</ReactMarkdown>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="empty-state" style={{ padding: '1rem', marginTop: '1rem' }}>
                        No saved plans yet. Generate one above to start tracking your study schedule!
                    </div>
                )}
            </div>
        </div>
    );
};

export default ActionPlanGenerator;
