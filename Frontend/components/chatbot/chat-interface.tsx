// chat-interface


"use client";
 
import React, { useState, useRef, useEffect } from 'react';
import { Message, Dataset } from '@/lib/types';
import { ChatMessage } from './chat-message';
import { ChatInput } from './chat-input';
import { DatasetGrid } from '@/components/datasets/dataset-grid';

import axios from 'axios';
import { useToast } from '@/hooks/use-toast';
<<<<<<< HEAD
 
=======
const API = process.env.NEXT_PUBLIC_SERVER_URL || 'http://localhost:8000/';

>>>>>>> 27e1dafe46b0ffab8040b3b7afd191c2e6f9d60c
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
  const { toast } = useToast();
  
 
  const handleSendMessage = async (content: string) => {
    const newUserMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content
    };
    setMessages(prev => [...prev, newUserMessage]);
    setIsProcessing(true);
 
    try {
      // Call backend API to search datasets using Groq
      const response = await axios.post(`${API}api/search/`, {
        query: content
      });
 
      const newAssistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `I've found some datasets related to "${content}". Here are my recommendations:`
      };
      setMessages(prev => [...prev, newAssistantMessage]);
 
      // Transform the API response to match our Dataset interface
      const formattedDatasets: Dataset[] = response.data.datasets.map((ds: any) => ({
        id: ds.ref,
        title: ds.title,
        description: ds.description,
        downloadLink: ds.url,
        externalLink: ds.url,
        owner: ds.owner,
        downloads: ds.downloads,
        lastUpdated: ds.lastUpdated,
        size: ds.size
      }));
 
      setDatasets(formattedDatasets);
 
      if (formattedDatasets.length === 0) {
        toast({
          title: "No datasets found",
          description: "Try a different search query",
          variant: "destructive",
        });
      }
 
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error while searching for datasets. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
      toast({
        title: "Error",
        description: "Failed to search datasets. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };
 
  const handleFileUpload = (file: File) => {
    
    // Add a message to show the uploaded file
    const fileMessage: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content: `File "${file.name}" has been uploaded. You can now proceed to the preprocessing tab to work with this dataset.`
    };
    setMessages(prev => [...prev, fileMessage]);
    toast({
      title: "File uploaded",
      description: `Your dataset "${file.name}" has been uploaded successfully. Switch to the preprocessing tab to continue.`,
      variant: "default",
    });
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
      {/* Restructured to have chat content at top and input fixed at bottom */}
<div className="flex-1 overflow-hidden flex flex-col">
        {/* Chat messages and dataset grid - takes up remaining space */}
<div className="flex-1 overflow-y-auto py-4 pb-20"> {/* Added bottom padding to ensure content isn't hidden behind input */}
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
</div>
      {/* Fixed input bar at bottom */}
<div className="fixed bottom-0 left-0 right-0 bg-background/80 backdrop-blur-sm z-10 border-t border-border/40">
<div className="max-w-5xl mx-auto px-4 py-4">
<ChatInput
            onSend={handleSendMessage}
            onFileUpload={handleFileUpload}
            isProcessing={isProcessing}
          />
</div>
</div>
</div>
  );
}