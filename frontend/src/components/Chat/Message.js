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

  const initials = message.role === 'user' ? 'U' : 'AI';

  return (
    <div className={`message-wrapper ${message.role}`}>
      <div className="message-avatar">
        {initials}
      </div>
      <div className="msg-body">
        <div className={`message-bubble${message.isError ? ' error' : ''}`}>
          <div
            className="message-content"
            dangerouslySetInnerHTML={{ __html: formatMessageText(message.content) }}
          />
          {message.role === 'assistant' && !message.isError && message.metadata && (
            <div className="message-metadata">
              {message.metadata.topic && (
                <span className="meta-tag">
                  <i className="fas fa-tag"></i>
                  {message.metadata.topic}
                  {message.metadata.subtopic && ` → ${message.metadata.subtopic}`}
                </span>
              )}
              {message.metadata.sourceCount && (
                <span className="meta-tag">
                  <i className="fas fa-book"></i>
                  {message.metadata.sourceCount} sources
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;