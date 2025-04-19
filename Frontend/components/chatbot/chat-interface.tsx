"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Message, Dataset } from '@/lib/types';
import { ChatMessage } from './chat-message';
import { ChatInput } from './chat-input';
import { DatasetGrid } from '@/components/datasets/dataset-grid';

// Simulated ML datasets for demo purposes
const MOCK_DATASETS: Dataset[] = [
  {
    id: '1',
    title: 'Iris Flower Classification Dataset',
    description: 'This is perhaps the best known database to be found in the pattern recognition literature. The dataset contains 3 classes of 50 instances each, where each class refers to a type of iris plant.',
    downloadLink: 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    externalLink: 'https://archive.ics.uci.edu/ml/datasets/iris',
    category: 'Classification'
  },
  {
    id: '2',
    title: 'Wine Quality Dataset',
    description: 'Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests.',
    downloadLink: 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    externalLink: 'https://archive.ics.uci.edu/ml/datasets/wine+quality',
    category: 'Regression'
  },
  {
    id: '3',
    title: 'Heart Disease Dataset',
    description: 'This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient.',
    downloadLink: 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
    externalLink: 'https://archive.ics.uci.edu/ml/datasets/heart+disease',
    category: 'Healthcare'
  },
];

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'What topic do you want to build a machine learning model on?'
    }
  ]);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSendMessage = (content: string) => {
    const newUserMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content
    };
    
    setMessages(prev => [...prev, newUserMessage]);
    setIsProcessing(true);
    
    // Simulate AI response after a delay
    setTimeout(() => {
      const newAssistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I've found some datasets related to "${content}". Here are my recommendations:`
      };
      
      setMessages(prev => [...prev, newAssistantMessage]);
      
      // Simulate dataset retrieval
      // In a real app, this would be an API call to the backend
      setTimeout(() => {
        setDatasets(MOCK_DATASETS);
        setIsProcessing(false);
      }, 1000);
    }, 1500);
  };

  const handleSelectDataset = (dataset: Dataset) => {
    // Here you would implement the logic for when a user selects a dataset
    console.log('Selected dataset:', dataset);
    // This would typically navigate to the next step in your application
  };

  // Auto-scroll to the latest message
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full max-w-5xl mx-auto px-4">
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto py-4">
          {messages.map((message) => (
            <ChatMessage 
              key={message.id}
              message={message}
              isLoading={isProcessing && message.id === messages[messages.length - 1].id}
            />
          ))}
          <div ref={messagesEndRef} />
          
          {datasets.length > 0 && (
            <div className="px-4 sm:px-10 lg:px-16">
              <DatasetGrid 
                datasets={datasets} 
                onSelectDataset={handleSelectDataset}
              />
            </div>
          )}
        </div>
        
        <div className="pt-4 pb-6">
          <ChatInput 
            onSend={handleSendMessage}
            isProcessing={isProcessing}
          />
        </div>
      </div>
    </div>
  );
}