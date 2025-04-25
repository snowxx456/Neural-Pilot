"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { RefreshCw, Play, Loader2 } from "lucide-react"
import { ModelCard } from "@/components/model_training/model-card"
import { ModelComparisonChart } from "@/components/model_training/model-comparison-chart"
import { FeatureImportanceChart } from "@/components/model_training/feature-importance-chart"
import { ConfusionMatrixChart } from "@/components/model_training/confusion-matrix-chart"
import { CorrelationMatrixChart } from "@/components/model_training/correlation-matrix-chart"
import { RocCurveChart } from "@/components/model_training/roc-curve-chart"
import { PrecisionRecallChart } from "@/components/model_training/precision-recall-chart"
import { TrainingProgress } from "@/components/model_training/training-progress"
import { fetchModelResults } from "@/lib/api"
import type { ModelResult } from "@/lib/types"
import { AnimatedBackground } from "@/components/model_training/animated-background"

export default function ModelsPage() {
  const [loading, setLoading] = useState(true)
  const [isTraining, setIsTraining] = useState(false)
  const [modelResults, setModelResults] = useState<ModelResult[]>([])
  const [activeTab, setActiveTab] = useState("models")

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        console.log("Fetching model results...")
        const data = await fetchModelResults()
        console.log("Model results fetched successfully:", data)
        setModelResults(data)
      } catch (error) {
        console.error("Failed to fetch model results:", error)
        // Show error message to user
        alert("Failed to load model results. Using mock data instead.")
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  const handleRefresh = async () => {
    try {
      setLoading(true)
      console.log("Refreshing model results...")
      const data = await fetchModelResults()
      console.log("Model results refreshed successfully:", data)
      setModelResults(data)
    } catch (error) {
      console.error("Failed to refresh model results:", error)
      // Show error message to user
      alert("Failed to refresh model results. Using previous data.")
    } finally {
      setLoading(false)
    }
  }

  const handleStartTraining = async () => {
    setIsTraining(true)
    // In a real implementation, you would call the API to start training
    // and then poll for updates or use WebSockets to get real-time progress
    
    // Simulate training completion after some time (remove this in production)
    setTimeout(() => {
      handleTrainingComplete()
    }, 10000)
  }

  const handleTrainingComplete = async () => {
    try {
      console.log("Training complete, fetching updated model results...")
      const data = await fetchModelResults()
      console.log("Updated model results fetched successfully:", data)
      setModelResults(data)
    } catch (error) {
      console.error("Failed to fetch updated model results:", error)
      alert("Training complete, but failed to fetch updated results.")
    } finally {
      setIsTraining(false)
    }
  }

  // Function to render the loading state for model cards
  const renderModelCardLoading = () => {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[...Array(6)].map((_, i) => (
          <Card key={i} className="overflow-hidden flex items-center justify-center h-64">
            <div className="flex flex-col items-center justify-center">
              <Loader2 className="h-12 w-12 text-primary animate-spin mb-4" />
              <p className="text-sm text-muted-foreground">Training models...</p>
            </div>
          </Card>
        ))}
      </div>
    )
  }

  // Function to render model cards or loading state
  const renderModelsContent = () => {
    if (isTraining) {
      return renderModelCardLoading()
    }
    
    if (loading) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <CardHeader className="pb-2">
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Skeleton className="h-20 w-full" />
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )
    }
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {modelResults.map((model) => (
          <ModelCard key={model.name} model={model} />
        ))}
      </div>
    )
  }

  // Function to render visualization content or loading state
  const renderVisualizationsContent = () => {
    if (isTraining || loading) {
      return (
        <div className="space-y-6">
          {[...Array(5)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-1/3" />
                <Skeleton className="h-4 w-1/2" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-[400px] w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      )
    }
    
    return (
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Feature Importance</CardTitle>
            <CardDescription>Top features contributing to the best model's predictions</CardDescription>
          </CardHeader>
          <CardContent>
            <FeatureImportanceChart />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Confusion Matrix</CardTitle>
            <CardDescription>Visualization of the best model's prediction accuracy</CardDescription>
          </CardHeader>
          <CardContent>
            <ConfusionMatrixChart />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Correlation Matrix</CardTitle>
            <CardDescription>Correlation between numerical features in the dataset</CardDescription>
          </CardHeader>
          <CardContent>
            <CorrelationMatrixChart />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>ROC Curve</CardTitle>
            <CardDescription>Receiver Operating Characteristic curve for classification models</CardDescription>
          </CardHeader>
          <CardContent>
            <RocCurveChart />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Precision-Recall Curve</CardTitle>
            <CardDescription>Precision vs Recall trade-off for classification models</CardDescription>
          </CardHeader>
          <CardContent>
            <PrecisionRecallChart />
          </CardContent>
        </Card>
      </div>
    )
  }

  // Function to render comparison content or loading state
  const renderComparisonContent = () => {
    if (isTraining || loading) {
      return (
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-1/3" />
            <Skeleton className="h-4 w-1/2" />
          </CardHeader>
          <CardContent className="pt-6">
            <Skeleton className="h-[400px] w-full" />
          </CardContent>
        </Card>
      )
    }
    
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Comparison</CardTitle>
          <CardDescription>Compare performance metrics across different models</CardDescription>
        </CardHeader>
        <CardContent className="pt-6">
          <ModelComparisonChart models={modelResults} />
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="min-h-screen bg-background relative">
      <AnimatedBackground />

      <div className="container py-10 relative z-10">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Model Results</h1>
            <p className="text-muted-foreground mt-1">
              View and compare the performance of trained machine learning models
            </p>
          </div>
          <div className="flex gap-2">
            <Button onClick={handleStartTraining} disabled={isTraining} className="gap-2">
              {isTraining ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Train Models
                </>
              )}
            </Button>
            <Button onClick={handleRefresh} variant="outline" className="gap-2" disabled={isTraining || loading}>
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>

        {isTraining && (
          <TrainingProgress />
        )}

        <Tabs defaultValue="models" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="comparison">Comparison</TabsTrigger>
            <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
          </TabsList>

          <TabsContent value="models" className="space-y-4">
            {renderModelsContent()}
          </TabsContent>

          <TabsContent value="comparison">
            {renderComparisonContent()}
          </TabsContent>

          <TabsContent value="visualizations" className="space-y-6">
            {renderVisualizationsContent()}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}