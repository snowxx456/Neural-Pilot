"use client"

import { useState, useEffect } from "react"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, Clock, AlertCircle } from "lucide-react"
import { estimateTrainingTime } from "@/lib/api"

interface TrainingProgressProps {
  isTraining: boolean
  datasetSize: number
  modelComplexity: "low" | "medium" | "high"
  onComplete?: () => void
}

export function TrainingProgress({ isTraining, datasetSize, modelComplexity, onComplete }: TrainingProgressProps) {
  const [progress, setProgress] = useState(0)
  const [currentModel, setCurrentModel] = useState("Preparing data...")
  const [timeRemaining, setTimeRemaining] = useState(0)
  const [error, setError] = useState<string | null>(null)

  // Models to simulate training
  const models = [
    "Preparing data...",
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "XGBoost",
    "SVM",
    "NaiveBayes",
    "Finalizing results...",
  ]

  useEffect(() => {
    if (!isTraining) {
      setProgress(0)
      setCurrentModel("Preparing data...")
      setTimeRemaining(0)
      setError(null)
      return
    }

    // Estimate total training time
    const totalTime = estimateTrainingTime(datasetSize, modelComplexity)
    setTimeRemaining(totalTime)

    // Reset progress
    setProgress(0)
    setCurrentModel(models[0])

    // Simulate training progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        // Calculate new progress
        const newProgress = prev + 100 / (totalTime * 10)

        // Update current model based on progress
        const modelIndex = Math.min(Math.floor((newProgress / 100) * models.length), models.length - 1)
        setCurrentModel(models[modelIndex])

        // Update time remaining
        const remainingPercentage = 1 - newProgress / 100
        setTimeRemaining(Math.max(0, Math.round(totalTime * remainingPercentage)))

        // Check if training is complete
        if (newProgress >= 100) {
          clearInterval(interval)
          if (onComplete) {
            setTimeout(onComplete, 500)
          }
          return 100
        }

        return Math.min(newProgress, 100)
      })
    }, 100)

    // Cleanup
    return () => clearInterval(interval)
  }, [isTraining, datasetSize, modelComplexity, onComplete])

  if (!isTraining) {
    return null
  }

  // Format time remaining
  const formatTimeRemaining = (seconds: number) => {
    if (seconds < 60) return `${seconds} seconds`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  }

  return (
    <Card className="mb-6">
      <CardHeader className="pb-3">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle className="text-lg flex items-center">
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Training Models
            </CardTitle>
            <CardDescription>Currently training: {currentModel}</CardDescription>
          </div>
          <div className="flex items-center text-sm text-muted-foreground">
            <Clock className="h-4 w-4 mr-1" />
            {timeRemaining > 0 ? (
              <span>Estimated time remaining: {formatTimeRemaining(timeRemaining)}</span>
            ) : (
              <span>Finalizing results...</span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <Progress value={progress} className="h-2" />

          {error && (
            <div className="flex items-center text-sm text-destructive">
              <AlertCircle className="h-4 w-4 mr-2" />
              {error}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
