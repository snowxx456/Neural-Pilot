"use client";

import React from 'react';
import { Dataset } from '@/lib/types';
import { DatasetCard } from './dataset-card';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useToast } from '@/hooks/use-toast';

interface DatasetGridProps {
  datasets: Dataset[];
  onSelectDataset: (dataset: Dataset) => void;
}

export function DatasetGrid({ datasets, onSelectDataset }: DatasetGridProps) {
  const { toast } = useToast();

  const handleDownload = async (e: React.MouseEvent, dataset: Dataset) => {
    e.stopPropagation();
    try {
      const response = await axios.get(dataset.url, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${dataset.title}.csv`);
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      link.remove();
      
      toast({
        title: "Success",
        description: "Dataset downloaded successfully",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to download dataset",
        variant: "destructive",
      });
    }
  };

  const handleSelect = async (dataset: Dataset) => {
    try {
      const response = await axios.post('http://localhost:8000/api/dataset/select/', {
        datasetRef: dataset.ref,
        url: dataset.url
      });

      toast({
        title: "Success",
        description: "Dataset selected for processing",
      });

      onSelectDataset({
        ...dataset,
        id: response.data.dataset_id
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to select dataset",
        variant: "destructive",
      });
    }
  };

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