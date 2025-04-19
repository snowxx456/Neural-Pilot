"use client";

import React, { useState, useEffect } from 'react';

interface TypingEffectProps {
  text: string;
  speed?: number;
  className?: string;
  onComplete?: () => void;
}

export function TypingEffect({ 
  text, 
  speed = 50, 
  className = '',
  onComplete
}: TypingEffectProps) {
  const [displayText, setDisplayText] = useState('');
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (index < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text.charAt(index));
        setIndex(prevIndex => prevIndex + 1);
      }, speed);
      
      return () => clearTimeout(timeout);
    } else if (onComplete) {
      onComplete();
    }
  }, [index, text, speed, onComplete]);

  return (
    <span className={className}>
      {displayText}
      {index < text.length && (
        <span className="inline-block w-[0.1em] h-[1em] bg-primary animate-blink ml-1"></span>
      )}
    </span>
  );
}