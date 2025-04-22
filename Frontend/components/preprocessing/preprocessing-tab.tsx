'use client';

import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowRight, 
  Database, 
  Download, 
  LineChart, 
  Loader2, 
  Sparkles,
  CheckCircle2,
  Binary,
  FileJson,
  FilterX,
  Scale,
  Sigma,
  Brain,
  Wand2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { PreprocessingStatus, PreprocessingStep } from '@/lib/types';
import { DataTable } from './data-table';
import { ProcessingStep } from './processing-step';
import { cn } from '@/lib/utils';

interface PreprocessingTabProps {
  datasetName: string;
  onVisualize: () => void;
}

const MOCK_STEPS: PreprocessingStep[] = [
  {
    id: 1,
    title: 'Loading Dataset',
    description: 'Reading and parsing the raw data file',
    icon: FileJson,
    status: 'pending'
  },
  {
    id: 2,
    title: 'Handling Missing Values',
    description: 'Identifying and imputing missing data points',
    icon: FilterX,
    status: 'pending'
  },
  {
    id: 3,
    title: 'Feature Scaling',
    description: 'Normalizing numerical features to a standard range',
    icon: Scale,
    status: 'pending'
  },
  {
    id: 4,
    title: 'Feature Engineering',
    description: 'Creating new features and transforming existing ones',
    icon: Binary,
    status: 'pending'
  },
  {
    id: 5,
    title: 'Statistical Analysis',
    description: 'Computing correlations and feature importance',
    icon: Sigma,
    status: 'pending'
  }
];

export function PreprocessingTab({ datasetName, onVisualize }: PreprocessingTabProps) {
  const [status, setStatus] = useState<PreprocessingStatus>('idle');
  const [sampleData, setSampleData] = useState<any[]>([]);
  const [steps, setSteps] = useState<PreprocessingStep[]>(MOCK_STEPS);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(-1);
  
  const updateStepStatus = useCallback((index: number, newStatus: 'pending' | 'processing' | 'completed') => {
    setSteps(prevSteps => 
      prevSteps.map((step, i) => ({
        ...step,
        status: i === index ? newStatus : step.status
      }))
    );
  }, []);

  const simulateProcessingStep = useCallback(async (index: number) => {
    setCurrentStepIndex(index);
    updateStepStatus(index, 'processing');
    await new Promise(resolve => setTimeout(resolve, 2000));
    updateStepStatus(index, 'completed');
  }, [updateStepStatus]);
  
  const handlePreprocess = async () => {
    try {
      setStatus('processing');
      setCurrentStepIndex(-1);
      setSteps(MOCK_STEPS);
      
      // Process steps sequentially
      for (let i = 0; i < steps.length; i++) {
        await simulateProcessingStep(i);
      }
      
      // Set sample data after all steps complete
      setSampleData([
        { id: 1, feature1: 'Value 1', feature2: 42, target: 1 },
        { id: 2, feature1: 'Value 2', feature2: 28, target: 0 },
        { id: 3, feature1: 'Value 3', feature2: 35, target: 1 },
      ]);
      
      setStatus('completed');
    } catch (error) {
      console.error('Preprocessing failed:', error);
      setStatus('error');
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 py-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-effect rounded-2xl p-8 mb-8 border border-primary/20 relative overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-chart-2/5 pointer-events-none" />
        
        <div className="flex items-center gap-6 mb-8">
          <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center group relative">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-chart-1/20 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
            <Brain className="h-8 w-8 text-primary group-hover:text-chart-1 transition-colors relative z-10" />
          </div>
          <div>
            <h2 className="text-2xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-primary via-chart-1 to-chart-2">
              AutoML Preprocessing
            </h2>
            <p className="text-lg text-muted-foreground flex items-center gap-2">
              <Database className="h-5 w-5" />
              {datasetName}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-4">
          <Button
            size="lg"
            onClick={handlePreprocess}
            disabled={status === 'processing'}
            className="gap-2 bg-gradient-to-r from-primary to-chart-1 hover:opacity-90 transition-opacity relative group overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
            {status === 'processing' ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Wand2 className="h-5 w-5" />
            )}
            <span className="relative z-10">Start Preprocessing</span>
          </Button>

          {status === 'completed' && (
            <>
              <Button
                variant="outline"
                size="lg"
                onClick={() => {/* Implement download logic */}}
                className="gap-2 border-chart-2/30 hover:border-chart-2/50 transition-colors relative group overflow-hidden"
              >
                <div className="absolute inset-0 bg-chart-2/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                <Download className="h-5 w-5 text-chart-2" />
                <span className="relative z-10">Download Cleaned Dataset</span>
              </Button>

              <Button
                variant="secondary"
                size="lg"
                onClick={onVisualize}
                className="gap-2 bg-chart-3/20 hover:bg-chart-3/30 transition-colors relative group overflow-hidden"
              >
                <div className="absolute inset-0 bg-chart-3/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                <LineChart className="h-5 w-5 text-chart-3" />
                <span className="relative z-10">Visualize Data</span>
              </Button>
            </>
          )}
        </div>
      </motion.div>

      <AnimatePresence>
        {status === 'processing' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-4 mb-8"
          >
            {steps.map((step, index) => (
              <ProcessingStep
                key={step.id}
                step={step}
                isActive={index === currentStepIndex}
                isCompleted={index < currentStepIndex}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {status === 'completed' && sampleData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="h-6 w-6 text-green-500" />
                <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-chart-1 to-chart-2">
                  Sample Cleaned Dataset
                </h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Showing first {sampleData.length} entries
              </p>
            </div>

            <div className="glass-effect rounded-xl overflow-hidden border border-primary/20 relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-chart-1/5 via-transparent to-chart-2/5 opacity-0 group-hover:opacity-100 transition-opacity" />
              <DataTable data={sampleData} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}