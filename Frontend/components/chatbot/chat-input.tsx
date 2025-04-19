"use client";

import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Send, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

interface ChatInputProps {
  onSend: (message: string) => void;
  isProcessing: boolean;
  placeholder?: string;
}

export function ChatInput({ 
  onSend, 
  isProcessing, 
  placeholder = "What topic do you want to build a machine learning model on?" 
}: ChatInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isProcessing) {
      onSend(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  return (
    <motion.form 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onSubmit={handleSubmit}
      className="relative w-full max-w-4xl mx-auto"
    >
      <div className="relative flex items-center">
        <div className="absolute left-4 top-1/2 -translate-y-1/2">
          <Sparkles className="h-4 w-4 text-muted-foreground" />
        </div>
        
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isProcessing}
          rows={1}
          className={cn(
            "resize-none w-full rounded-xl py-3 pl-11 pr-14",
            "glass-effect focus:ring-1 focus:ring-primary",
            "text-sm transition-all",
            "focus:outline-none",
            isProcessing && "opacity-70"
          )}
        />
        
        <Button
          type="submit"
          size="icon"
          disabled={!input.trim() || isProcessing}
          className={cn(
            "absolute right-1.5 h-9 w-9 rounded-lg",
            "bg-primary hover:bg-primary/90",
            "transition-all",
            (!input.trim() || isProcessing) && "opacity-70"
          )}
        >
          <Send className="h-4 w-4" />
          <span className="sr-only">Send</span>
        </Button>
      </div>
    </motion.form>
  );
}