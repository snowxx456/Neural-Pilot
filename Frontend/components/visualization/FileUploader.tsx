'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { FileText, Upload, AlertCircle, Brain, ChartBar, Database } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { DatasetType } from '@/lib/types'
import { analyzeCSV } from '@/lib/dataService'

interface FileUploaderProps {
  onDatasetReady: (dataset: DatasetType) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
}

export function FileUploader({ 
  onDatasetReady, 
  isLoading, 
  setIsLoading, 
  error, 
  setError 
}: FileUploaderProps) {
  const [progress, setProgress] = useState(0)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  const processFile = async (file: File) => {
    setIsLoading(true)
    setError(null)
    setProgress(10)
    
    try {
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 300)
      
      const result = await analyzeCSV(file)
      clearInterval(progressInterval)
      setProgress(100)
      onDatasetReady(result)
    } catch (err) {
      setError(`Failed to process file: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setIsLoading(false)
    }
  }
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file && file.type === 'text/csv') {
      setUploadedFile(file)
      processFile(file)
    } else {
      setError('Please upload a valid CSV file')
    }
  }, [onDatasetReady, setError, setIsLoading])
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    maxFiles: 1,
    disabled: isLoading
  })
  
  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="text-center space-y-4 mb-8">
        <div className="flex items-center justify-center gap-3">
          <Brain className="h-8 w-8 text-primary animate-pulse" />
          <h2 className="text-3xl font-bold">AutoML Data Visualization</h2>
        </div>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Upload your dataset and let our AI analyze and create intelligent visualizations. 
          Our system automatically detects patterns, relationships, and insights in your data.
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card className="glass neon-border">
          <CardContent className="p-6 text-center">
            <Database className="h-8 w-8 mx-auto mb-3 text-chart-1" />
            <h3 className="font-semibold mb-2">Smart Data Analysis</h3>
            <p className="text-sm text-muted-foreground">
              Automatic data type detection and statistical analysis
            </p>
          </CardContent>
        </Card>
        
        <Card className="glass neon-border">
          <CardContent className="p-6 text-center">
            <Brain className="h-8 w-8 mx-auto mb-3 text-chart-2" />
            <h3 className="font-semibold mb-2">AI-Powered Insights</h3>
            <p className="text-sm text-muted-foreground">
              Intelligent chart recommendations based on your data
            </p>
          </CardContent>
        </Card>
        
        <Card className="glass neon-border">
          <CardContent className="p-6 text-center">
            <ChartBar className="h-8 w-8 mx-auto mb-3 text-chart-3" />
            <h3 className="font-semibold mb-2">Interactive Visualization</h3>
            <p className="text-sm text-muted-foreground">
              Create and customize beautiful, interactive charts
            </p>
          </CardContent>
        </Card>
      </div>
      
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <Card 
        className={`upload-area glass neon-border ${
          isDragActive ? 'border-primary/50 bg-primary/5' : ''
        }`}
      >
        <CardContent className="p-0">
          <div
            {...getRootProps()}
            className={`flex flex-col items-center justify-center p-12 text-center cursor-pointer ${
              isLoading ? 'opacity-50 pointer-events-none' : ''
            }`}
          >
            <input {...getInputProps()} />
            
            <div className="mb-6 rounded-full bg-primary/10 p-6">
              {isLoading ? (
                <FileText className="h-12 w-12 text-primary animate-pulse" />
              ) : (
                <Upload className="h-12 w-12 text-primary" />
              )}
            </div>
            
            <div className="space-y-2">
              <p className="text-lg font-medium">
                {isDragActive
                  ? "Drop your dataset here"
                  : isLoading
                  ? "Analyzing your data..."
                  : "Drag & drop your dataset here"}
              </p>
              <p className="text-sm text-muted-foreground">
                {isLoading
                  ? uploadedFile?.name
                  : "Upload a CSV file to begin your data exploration journey"}
              </p>
            </div>
            
            {isLoading && (
              <div className="w-full mt-8 space-y-2 max-w-md">
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-muted-foreground">
                  {progress < 30 
                    ? "Reading and validating data..." 
                    : progress < 60 
                    ? "Detecting patterns and relationships..." 
                    : progress < 90 
                    ? "Generating intelligent visualizations..." 
                    : "Finalizing analysis..."}
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}