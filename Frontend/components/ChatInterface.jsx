import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HiChevronRight, HiOutlineLightningBolt, HiOutlineSearch } from 'react-icons/hi';
import DatasetCard from './DatasetCard';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      sender: 'bot', 
      text: 'Welcome to NEORO PILOT! What topic do you want to build a machine learning model on?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [datasets, setDatasets] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: messages.length + 1,
      sender: 'user',
      text: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/search-dataset', { // Updated path
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) {
        throw new Error(await response.text() || 'Failed to fetch datasets');
      }

      const data = await response.json();
      
      const botResponse = {
        id: messages.length + 2,
        sender: 'bot',
        text: `I've found ${data.datasets.length} datasets related to "${input}". Please select one to begin building your ML model:`,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botResponse]);
      setDatasets(data.datasets);
    } catch (error) {
      console.error('Error:', error);
      const errorResponse = {
        id: messages.length + 2,
        sender: 'bot',
        text: `Sorry, I encountered an error searching for datasets: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex flex-col h-full w-full max-w-4xl mx-auto">
      {/* Chat Messages Area */}
      <div className="flex-grow overflow-y-auto mb-4 pb-4 px-4 scrollbar-thin">
        <div className="space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className={`chat-bubble ${message.sender === 'user' ? 'user-bubble' : 'bot-bubble'} glass-effect`}
              >
                <p className="text-white">{message.text}</p>
                <span className="text-xs text-neo-light block text-right mt-1">
                  {formatTime(message.timestamp)}
                </span>
              </motion.div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="chat-bubble bot-bubble glass-effect"
              >
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 rounded-full bg-neo-primary animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-neo-secondary animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-neo-accent animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </motion.div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Datasets Display */}
      <AnimatePresence>
        {datasets.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="w-full mb-6"
          >
            <h3 className="text-lg font-medium text-white mb-3 pl-4">Available Datasets</h3>
            <div className="flex overflow-x-auto pb-4 gap-4 pl-4 pr-4 scrollbar-thin">
              {datasets.map((dataset) => (
                <DatasetCard 
                  key={dataset.ref} 
                  dataset={{
                    ...dataset,
                    downloadUrl: `#download-${dataset.ref}`,
                    externalLink: dataset.url
                  }}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input Area */}
      <div className="bg-neo-darker p-4 rounded-lg glass-effect">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <div className="relative flex-grow">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="What topic do you want to build a machine learning model on?"
              className="w-full bg-neo-dark text-white rounded-lg py-3 pl-4 pr-10 focus:outline-none focus:ring-2 focus:ring-neo-primary border border-neo-dark"
            />
            <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-neo-light">
              <HiOutlineSearch size={20} />
            </span>
          </div>
          <button
            type="submit"
            className="bg-neo-primary hover:bg-opacity-90 text-white p-3 rounded-lg flex-shrink-0 transition-all"
            disabled={isLoading}
          >
            <HiChevronRight size={20} />
          </button>
        </form>
        <div className="flex items-center mt-3 text-xs text-neo-light px-2">
          <HiOutlineLightningBolt className="text-neo-secondary mr-2" />
          <span>NEORO PILOT uses AutoML to instantly generate ML models from your selected datasets</span>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;