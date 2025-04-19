"use client";

import React from 'react';

interface AnimatedGradientProps {
  className?: string;
}

export function AnimatedGradient({ className = '' }: AnimatedGradientProps) {
  return (
    <div className={`absolute inset-0 overflow-hidden ${className}`}>
      <div className="absolute -inset-[10px] opacity-30">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] rounded-full bg-gradient-to-r from-chart-1 via-chart-2 to-chart-3 blur-[60px] animate-slow-spin" />
        <div className="absolute top-3/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] rounded-full bg-gradient-to-r from-chart-4 to-chart-5 blur-[70px] animate-slow-spin-reverse" />
      </div>
    </div>
  );
}