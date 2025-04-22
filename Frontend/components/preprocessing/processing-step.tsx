'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { PreprocessingStep } from '@/lib/types';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProcessingStepProps {
  step: PreprocessingStep;
  isActive: boolean;
  isCompleted: boolean;
}

export function ProcessingStep({ step, isActive, isCompleted }: ProcessingStepProps) {
  const Icon = step.icon;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className={cn(
        "glass-effect rounded-xl p-4 border transition-colors duration-300",
        isActive ? "border-primary/30 bg-primary/5" : "border-border/30",
        isCompleted && "border-chart-2/30 bg-chart-2/5"
      )}
    >
      <div className="flex items-center gap-4">
        <div className={cn(
          "h-12 w-12 rounded-lg flex items-center justify-center transition-colors duration-300",
          isActive ? "bg-primary/20" : "bg-muted",
          isCompleted && "bg-chart-2/20"
        )}>
          {isActive ? (
            <Loader2 className="h-6 w-6 text-primary animate-spin" />
          ) : (
            <Icon className={cn(
              "h-6 w-6 transition-colors duration-300",
              isCompleted ? "text-chart-2" : "text-muted-foreground"
            )} />
          )}
        </div>
        
        <div>
          <h4 className={cn(
            "text-lg font-medium transition-colors duration-300",
            isActive && "text-primary",
            isCompleted && "text-chart-2"
          )}>
            {step.title}
          </h4>
          <p className="text-sm text-muted-foreground">
            {step.description}
          </p>
        </div>
      </div>
    </motion.div>
  );
}