import React, { useState, useRef, useEffect } from 'react';
import Header from '../Layout/Header';
import Sidebar from '../Layout/Sidebar';
import Message from './Message';
import WelcomeMessage from './WelcomeMessage';
import { chatAPI } from '../../services/api';
import './ChatInterface.css';

const ChatInterface = ({ user, onLogout }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [topics, setTopics] = useState([]);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    loadTopics();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadTopics = async () => {
    try {
      const response = await chatAPI.getTopics();
      setTopics(response.data.topics || []);
    } catch (error) {
      console.error('Failed to load topics:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await chatAPI.query(input);
      const assistantMessage = { 
        role: 'assistant', 
        content: response.data.answer,
        metadata: {
          topic: response.data.detected_topic,
          subtopic: response.data.detected_subtopic,
          sourceCount: response.data.source_count
        },
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.',
        isError: true,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickQuestion = (question) => {
    setInput(question);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-interface-container">
      <Header 
        user={user} 
        onLogout={onLogout} 
        title="Technical Interview" 
        showBack={true}
      />

      <div className="chat-main-content">
        <Sidebar 
          topics={topics}
          onQuickQuestion={handleQuickQuestion}
        />

        <section className="chat-section">
          <div className="chat-container">
            <div className="chat-messages">
              {messages.length === 0 ? (
                <WelcomeMessage />
              ) : (
                messages.map((message, index) => (
                  <Message key={index} message={message} />
                ))
              )}
              {loading && (
                <div className="typing-indicator">
                  <i className="fas fa-circle"></i>
                  <i className="fas fa-circle"></i>
                  <i className="fas fa-circle"></i>
                  AI is thinking...
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-container">
              <div className="chat-input-wrapper">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about DBMS, OOPs, or OS..."
                  disabled={loading}
                  className="chat-input"
                />
                <button 
                  onClick={handleSendMessage}
                  disabled={!input.trim() || loading}
                  className="send-btn"
                >
                  <i className="fas fa-paper-plane"></i>
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default ChatInterface;