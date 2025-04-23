"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { RefreshCw, Play } from "lucide-react"
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
  }

  const handleTrainingComplete = async () => {
    setIsTraining(false)
    await handleRefresh()
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
              <Play className="h-4 w-4" />
              Train Models
            </Button>
            <Button onClick={handleRefresh} variant="outline" className="gap-2" disabled={isTraining}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </div>
        </div>

        {isTraining && (
          <TrainingProgress
            isTraining={isTraining}
            datasetSize={2000}
            modelComplexity="medium"
            onComplete={handleTrainingComplete}
          />
        )}

        <Tabs defaultValue="models" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          <TabsList className="grid w-full max-w-md grid-cols-3">
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="comparison">Comparison</TabsTrigger>
            <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
          </TabsList>

          <TabsContent value="models" className="space-y-4">
            {loading ? (
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
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {modelResults.map((model) => (
                  <ModelCard key={model.name} model={model} />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="comparison">
            <Card>
              <CardHeader>
                <CardTitle>Model Comparison</CardTitle>
                <CardDescription>Compare performance metrics across different models</CardDescription>
              </CardHeader>
              <CardContent className="pt-6">
                {loading ? <Skeleton className="h-[400px] w-full" /> : <ModelComparisonChart models={modelResults} />}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="visualizations" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
                <CardDescription>Top features contributing to the best model's predictions</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? <Skeleton className="h-[400px] w-full" /> : <FeatureImportanceChart />}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix</CardTitle>
                <CardDescription>Visualization of the best model's prediction accuracy</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? <Skeleton className="h-[400px] w-full" /> : <ConfusionMatrixChart />}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Correlation Matrix</CardTitle>
                <CardDescription>Correlation between numerical features in the dataset</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? <Skeleton className="h-[400px] w-full" /> : <CorrelationMatrixChart />}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>ROC Curve</CardTitle>
                <CardDescription>Receiver Operating Characteristic curve for classification models</CardDescription>
              </CardHeader>
              <CardContent>{loading ? <Skeleton className="h-[400px] w-full" /> : <RocCurveChart />}</CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Precision-Recall Curve</CardTitle>
                <CardDescription>Precision vs Recall trade-off for classification models</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? <Skeleton className="h-[400px] w-full" /> : <PrecisionRecallChart />}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
