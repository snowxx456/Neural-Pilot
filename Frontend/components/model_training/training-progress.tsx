"use client";

import { useState, useEffect } from "react";
import { Progress } from "@/components/ui/progress";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Loader2, Clock, AlertCircle, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { list } from "postcss";
import { fetchModelResults } from "@/lib/api";

let id: number | null = null;
let name: string | null = null;
const storedData = localStorage.getItem("selectedDataset");
if (storedData) {
  try {
    const data = JSON.parse(storedData);
    id = data.id;
    name = data.name;
    console.log("Dataset ID:", data.id);
  } catch (error) {
    console.error("Error parsing dataset from localStorage:", error);
  }
}

interface StepDetails {
  [key: string]: any;
}

interface TrainingStep {
  id: number;
  name: string;
  status: "pending" | "processing" | "completed" | "error";
  details: StepDetails;
}

interface TrainingProgressProps {
  startTraining: () => Promise<void>;
  isTraining: boolean;
  onTrainingComplete: () => Promise<void>;
  datasetId: number | null;
  datasetName: string | null;
}

export function TrainingProgress({
  startTraining,
  isTraining,
  onTrainingComplete,
  datasetId,
  datasetName,
}: TrainingProgressProps) {
  const [progress, setProgress] = useState(0);
  const [currentModel, setCurrentModel] = useState("Preparing data...");
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [steps, setSteps] = useState<{ [key: number]: TrainingStep }>({});
  const [isDownloading, setIsDownloading] = useState(false);

  // Make sure the API URL has the correct format
  const API = (
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/"
  ).replace(/\/?$/, "/");
  const Spinner = ({ className = "" }) => (
    <div
      className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent ${className}`}
    ></div>
  );
  useEffect(() => {
    const loadData = async () => {
      console.log("Fetching model results...");
      const data = await fetchModelResults();
      if (data.length !== 0) {
        setTrainingComplete(true);
      }
    };

    loadData();
  }, []);

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
  ];

  // Add to log function
  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLog((prev) => [...prev, `[${timestamp}] ${message}`]);
    console.log(`[${timestamp}] ${message}`); // Add console.log for debugging
  };

  // Download trained model function
  const downloadModel = async () => {
    try {
      // Show loading indicator or notification
      setIsDownloading(true);
      addLog(`Initiating download for model: ${id}`);

      // Check if we have a valid model ID
      if (!id) {
        throw new Error("Model ID is required for download");
      }

      // Construct the download URL
      const downloadUrl = `${API}api/download-model/${id}/`;

      // For smaller files, you could use fetch:
      // const response = await fetch(downloadUrl);
      // if (!response.ok) throw new Error(`Download failed: ${response.statusText}`);
      // const blob = await response.blob();
      // const url = window.URL.createObjectURL(blob);

      // For simplicity and better handling of large files, direct browser download:
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.setAttribute("download", `model-${id}.pkl`); // Default name, server will override
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      addLog(`Download started for model: ${id}`);
      return true;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      console.error("Error downloading model:", error);
      addLog(`Error downloading model: ${errorMessage}`);

      return false;
    } finally {
      // Hide loading indicator
      setIsDownloading(false);
    }
  };

  // Connect to SSE stream
  useEffect(() => {
    if (!isTraining) return;

    console.log("Setting up SSE stream"); // Debug log
    let eventSource: EventSource | null = null;
    let retryCount = 0;
    const maxRetries = 10;
    const baseRetryDelay = 1000; // 1 second

    const connectSSE = () => {
      // Close any existing connection first
      if (eventSource) {
        eventSource.close();
      }

      eventSource = new EventSource(`${API}api/stream/`);

      eventSource.onopen = () => {
        setConnected(true);
        retryCount = 0; // Reset retry count on successful connection
        addLog("Connected to SSE stream");
      };

      eventSource.onerror = () => {
        if (eventSource) {
          eventSource.close();
        }
        setConnected(false);

        if (retryCount < maxRetries) {
          retryCount++;
          const delay = Math.min(
            30000,
            baseRetryDelay * Math.pow(2, retryCount)
          ); // Exponential backoff
          addLog(
            `SSE connection error. Retrying in ${
              delay / 1000
            } seconds... (${retryCount}/${maxRetries})`
          );

          setTimeout(connectSSE, delay);
        } else {
          addLog("Maximum retry attempts reached. Please refresh the page.");
        }
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TrainingStep;

          setSteps((prev) => ({
            ...prev,
            [data.id]: data,
          }));

          // Log the step change
          const statusEmoji =
            data.status === "completed"
              ? "✅"
              : data.status === "processing"
              ? "⏳"
              : data.status === "error"
              ? "❌"
              : "⏱️";

          addLog(
            `${statusEmoji} Step ${data.id} (${data.name}): ${data.status}`
          );

          // Update progress based on steps
          const stepProgress = (data.id / 9) * 100;
          setProgress(stepProgress);

          // Update current model based on the step
          const modelIndex = Math.min(
            Math.floor((stepProgress / 100) * models.length),
            models.length - 1
          );
          setCurrentModel(models[modelIndex]);

          // Update time remaining (a simple simulation)
          const remainingPercentage = 1 - stepProgress / 100;
          setTimeRemaining(Math.max(0, Math.round(300 * remainingPercentage))); // 300 seconds as example

          // Log details if present
          if (data.details && Object.keys(data.details).length > 0) {
            const detailsStr = Object.entries(data.details)
              .map(([key, value]) => `${key}: ${value}`)
              .join(", ");

            if (detailsStr) {
              addLog(`   Details: ${detailsStr}`);
            }
          }

          // Check if this is the final step and it's completed
          if (data.id === 9 && data.status === "completed") {
            setTrainingComplete(true);
            onTrainingComplete(); // Call parent callback when training completes

            // Save model info to localStorage
          }
        } catch (error) {
          addLog(
            `Error parsing SSE data: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
        }
      };
    };

    connectSSE();

    // Implement a periodic status checking mechanism if SSE disconnects
    const statusCheckInterval = setInterval(() => {
      if (isTraining && !connected && datasetId) {
        // If we're training but not connected to SSE, fetch status another way
        fetch(`${API}api/training-status/${datasetId}/`)
          .then((response) => {
            if (response.ok) return response.json();
            throw new Error("Failed to fetch training status");
          })
          .then((data) => {
            // Update local state with server state
            if (data.steps) {
              setSteps(data.steps);
            }
            if (data.completed) {
              setTrainingComplete(data.completed);
              onTrainingComplete();
            }
            addLog("Retrieved training status via fallback endpoint");
          })
          .catch((err) => {
            console.error("Failed to check training status:", err);
          });
      }
    }, 10000); // Check every 10 seconds

    // Cleanup function
    return () => {
      clearInterval(statusCheckInterval);
      if (eventSource) {
        eventSource.close();
        addLog("Disconnected from SSE stream");
      }
    };
  }, [datasetId, datasetName, isTraining, connected, API, onTrainingComplete]);

  // Reset progress when training is started or stopped
  useEffect(() => {
    if (!isTraining) {
      setProgress(0);
      setCurrentModel("Preparing data...");
      setTimeRemaining(0);
      setError(null);
    }
  }, [isTraining]);

  // Format time remaining
  const formatTimeRemaining = (seconds: number) => {
    if (seconds < 60) return `${seconds} seconds`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <Card className="mb-6">
      <CardHeader className="pb-3">
        <div className="flex justify-between items-center">
          <div>
            <CardTitle className="text-lg flex items-center">
              {isTraining ? (
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              ) : (
                <Play className="h-5 w-5 mr-2" />
              )}
              {isTraining ? "Training Models" : "Model Training"}
            </CardTitle>
            <CardDescription>
              {isTraining
                ? `Currently training: ${currentModel}`
                : "Start training to build your model"}
            </CardDescription>
          </div>
          <div className="flex items-center text-sm text-muted-foreground">
            {isTraining && timeRemaining > 0 ? (
              <div className="flex items-center">
                <Clock className="h-4 w-4 mr-1" />
                <span>
                  Estimated time remaining: {formatTimeRemaining(timeRemaining)}
                </span>
              </div>
            ) : isTraining ? (
              <div className="flex items-center">
                <Clock className="h-4 w-4 mr-1" />
                <span>Finalizing results...</span>
              </div>
            ) : (
              <div className="flex items-center">
                <span>Dataset: {datasetName || "None selected"}</span>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {isTraining ? (
            <Progress value={progress} className="h-2" />
          ) : (
            <div className="flex justify-between gap-4">
              <Button
                variant="default"
                onClick={startTraining}
                disabled={!datasetId}
                className="flex-1"
              >
                Start Training
              </Button>

              {trainingComplete && (
                <Button
                  variant="outline"
                  onClick={() => downloadModel()}
                  disabled={isDownloading}
                  className="flex-1"
                >
                  {isDownloading ? (
                    <>
                      <Spinner className="mr-2 h-4 w-4 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    "Download Trained Model"
                  )}
                </Button>
              )}
            </div>
          )}

          {error && (
            <div className="flex items-center text-sm text-destructive">
              <AlertCircle className="h-4 w-4 mr-2" />
              {error}
            </div>
          )}

          {log.length > 0 && (
            <div className="mt-4 border rounded-md p-3 bg-muted/50 h-32 overflow-y-auto">
              <h4 className="text-sm font-medium mb-2">Training Log</h4>
              <div className="space-y-1 text-xs font-mono">
                {log.map((entry, index) => (
                  <div key={index} className="text-muted-foreground">
                    {entry}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
