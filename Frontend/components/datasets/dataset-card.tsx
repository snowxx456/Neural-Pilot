"use client";

import React from 'react';
import { cn } from '@/lib/utils';
import { Dataset } from '@/lib/types';
import { Download, ExternalLink, FileText, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';

interface DatasetCardProps {
  dataset: Dataset;
  onClick: () => void;
}

export function DatasetCard({ dataset, onClick }: DatasetCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className={cn(
        "relative flex flex-col h-full overflow-hidden",
        "glass-effect rounded-xl",
        "transition-all hover:border-primary/20 cursor-pointer",
        "group"
      )}
      onClick={onClick}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-chart-1/5 via-transparent to-chart-2/5 opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-chart-1 to-chart-2 transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />
      
      <div className="p-6 relative">
        <div className="mb-4 flex items-center justify-between">
          <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center group-hover:scale-110 transition-transform">
            <Sparkles className="h-6 w-6 text-primary animate-pulse-slow" />
          </div>
          
          <div className="px-3 py-1.5 rounded-full glass-effect text-chart-2 text-xs font-medium border border-chart-2/20">
            {dataset.category || 'Dataset'}
          </div>
        </div>
        
        <h3 className="text-lg font-medium mb-2 line-clamp-1 group-hover:text-glow transition-all">
          {dataset.title}
        </h3>
        
        <p className="text-sm text-muted-foreground mb-6 line-clamp-3">
          {dataset.description}
        </p>
        
        <div className="mt-auto flex gap-2">
          <Button 
            variant="outline" 
            size="sm"
            className="flex-1 gap-1.5 group/btn"
            onClick={(e) => {
              e.stopPropagation();
              window.open(dataset.externalLink, '_blank');
            }}
          >
            <ExternalLink className="h-4 w-4 group-hover/btn:text-chart-1 transition-colors" />
            <span>View Details</span>
          </Button>
          
          <Button 
            variant="secondary" 
            size="sm"
            className="flex-1 gap-1.5 group/btn"
            onClick={(e) => {
              e.stopPropagation();
              window.open(dataset.downloadLink, '_blank');
            }}
          >
            <Download className="h-4 w-4 group-hover/btn:text-chart-2 transition-colors" />
            <span>Download</span>
          </Button>
        </div>
      </div>
    </motion.div>
  );
}