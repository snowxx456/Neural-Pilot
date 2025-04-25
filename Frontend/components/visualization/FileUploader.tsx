"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  FileText,
  Upload,
  AlertCircle,
  Brain,
  ChartBar,
  Database,
  Loader2,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { DatasetType } from "@/lib/types";
import { analyzeCSV } from "@/lib/dataService";
import { Button } from "@/components/ui/button";

const API = process.env.NEXT_PUBLIC_SERVER_URL || "http://localhost:8000/";

interface FileUploaderProps {
  datasetId: any;
  onDatasetReady: (dataset: DatasetType) => void;
  isLoading: boolean;
  datasetname: string;
  setIsLoading: (loading: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
}

export function FileUploader({
  datasetId,
  onDatasetReady,
  isLoading,
  setIsLoading,
  error,
  setError,
  datasetname,
}: FileUploaderProps) {
  const [progress, setProgress] = useState(0);

  const startVisualization = async () => {
    if (!datasetId) {
      setError("No dataset ID provided");
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      setProgress(10);

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 300);

      const response = await fetch(`${API}api/visualization/${datasetId}/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        throw new Error("Failed to start visualization");
      }

      const data = await response.json();
      setProgress(100);
      onDatasetReady(data);
    } catch (err) {
      setError(
        `Failed to start visualization: ${
          err instanceof Error ? err.message : "Unknown error"
        }`
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="text-center space-y-4 mb-8">
        <div className="flex items-center justify-center gap-3">
          <Brain className="h-8 w-8 text-primary animate-pulse" />
          <h2 className="text-3xl font-bold">AutoML Data Visualization</h2>
        </div>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Let our AI analyze and create intelligent visualizations for your
          dataset. Our system automatically detects patterns, relationships, and
          insights in your data.
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
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card className="glass neon-border">
        <CardContent className="flex flex-col items-center justify-center p-10 text-center">
          <div className="w-full text-center mb-4">
            <h3 className="text-lg font-medium mb-2">
              Dataset Ready for Visualization
            </h3>
            <p className="text-sm text-muted-foreground">
              Click the button below to analyze and visualize dataset{" "}
              {datasetname}
            </p>
          </div>

          <Button
            className="w-full glass neon-border bg-primary/10 hover:bg-primary/20"
            onClick={startVisualization}
            disabled={isLoading || !datasetId}
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Processing... {progress}%
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <ChartBar className="h-4 w-4" />
                Start Data Visualization
              </div>
            )}
          </Button>

          {progress > 0 && progress < 100 && (
            <div className="w-full mt-4">
              <div className="w-full bg-secondary/20 rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
