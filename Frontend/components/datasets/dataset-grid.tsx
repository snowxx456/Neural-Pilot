"use client";

import React from 'react';
import { Dataset } from '@/lib/types';
import { DatasetCard } from './dataset-card';
import { motion } from 'framer-motion';

interface DatasetGridProps {
  datasets: Dataset[];
  onSelectDataset: (dataset: Dataset) => void;
}

export function DatasetGrid({ datasets, onSelectDataset }: DatasetGridProps) {
  if (!datasets.length) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="w-full mt-8"
    >
      <h2 className="text-xl font-medium mb-6">Recommended Datasets</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-auto pb-2">
        {datasets.map((dataset, index) => (
          <DatasetCard
            key={dataset.id || index}
            dataset={dataset}
            onClick={() => onSelectDataset(dataset)}
          />
        ))}
      </div>
    </motion.div>
  );
}