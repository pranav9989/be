import React from 'react';
import './Message.css';

const Message = ({ message }) => {
  const formatMessageText = (text) => {
    return text
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>');
  };

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        <i className={`fas ${message.role === 'user' ? 'fa-user' : 'fa-robot'}`}></i>
      </div>
      <div className="message-content">
        <div 
          className="message-text"
          dangerouslySetInnerHTML={{ __html: formatMessageText(message.content) }}
        />
        {message.role === 'assistant' && !message.isError && message.metadata && (
          <div className="message-meta">
            {message.metadata.topic && (
              <span className="topic-badge">
                {message.metadata.topic}
                {message.metadata.subtopic && ` → ${message.metadata.subtopic}`}
              </span>
            )}
            {message.metadata.sourceCount && (
              <div className="sources-info">
                <i className="fas fa-books"></i> Based on {message.metadata.sourceCount} knowledge sources
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Message;