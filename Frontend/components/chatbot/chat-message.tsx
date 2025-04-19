import React from 'react';
import { cn } from '@/lib/utils';
import { Message } from '@/lib/types';
import { User, Bot } from 'lucide-react';
import { TypingEffect } from '@/components/ui/typing-effect';
import { motion } from 'framer-motion';

interface ChatMessageProps {
  message: Message;
  isLoading?: boolean;
}

export function ChatMessage({ message, isLoading = false }: ChatMessageProps) {
  const isUser = message.role === 'user';
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex w-full py-4",
        isUser ? "justify-end pl-10" : "justify-start pr-10"
      )}
    >
      <div 
        className={cn(
          "flex gap-3 max-w-3xl",
          isUser ? "flex-row-reverse" : "flex-row"
        )}
      >
        <div className={cn(
          "flex-shrink-0 h-10 w-10 rounded-xl flex items-center justify-center",
          isUser ? "bg-primary/20" : "bg-chart-2/20",
          "transition-transform hover:scale-105"
        )}>
          {isUser ? (
            <User className="h-5 w-5 text-primary" />
          ) : (
            <Bot className="h-5 w-5 text-chart-2" />
          )}
        </div>
        
        <div className={cn(
          "rounded-2xl py-3 px-4",
          isUser 
            ? "glass-effect bg-primary/10" 
            : "glass-effect bg-card/50",
          "transition-all hover:border-primary/20"
        )}>
          {isLoading && !isUser ? (
            <TypingEffect text={message.content} speed={20} className="text-sm" />
          ) : (
            <p className="text-sm">{message.content}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
}