"use client";

import React, { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
  Wand2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { PreprocessingStatus, PreprocessingStep } from "@/lib/types";
import { DataTable } from "./data-table";
import { ProcessingStep } from "./processing-step";
const API_URL = process.env.NEXT_PUBLIC_SERVER_URL || "http://localhost:8000/";
import { useRouter } from "next/navigation";

interface PreprocessingTabProps {
  datasetName: string;
}

const INITIAL_STEPS: PreprocessingStep[] = [
  {
    id: 1,
    title: "Loading Dataset",
    description: "Reading and parsing the raw data file",
    icon: FileJson,
    status: "pending",
  },
  {
    id: 2,
    title: "Handling Index Columns",
    description: "Identifying and Index columns and removing it",
    icon: FilterX,
    status: "pending",
  },
  {
    id: 3,
    title: "Handling Missing Values",
    description: "Identifying missing values and applying appropiate methods",
    icon: Scale,
    status: "pending",
  },
  {
    id: 4,
    title: "Handling Outliers",
    description: "Finding outliers and normalizing the data",
    icon: Binary,
    status: "pending",
  },
  {
    id: 5,
    title: "Removing duplicate columns",
    description: "Removing duplicate columns from the dataset",
    icon: Sigma,
    status: "pending",
  },
  {
    id: 6,
    title: "Saving the cleaned dataset",
    description: "Saving the cleaned dataset to a file",
    icon: Sparkles,
    status: "pending",
  },
];

export function PreprocessingTab({ datasetName }: PreprocessingTabProps) {
  const [status, setStatus] = useState<PreprocessingStatus>("idle");
  const [sampleData, setSampleData] = useState<any[]>([]);
  const [steps, setSteps] = useState<PreprocessingStep[]>(INITIAL_STEPS);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(-1);
  const [sseConnected, setSseConnected] = useState(false);
  const [sseError, setSseError] = useState<string | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [datasetId, setDatasetId] = useState<any | null>(null);
  const [name, setDatasetName] = useState<string>("");
  const router = useRouter();

  // Load dataset info from localStorage once on component mount
  useEffect(() => {
    const storedData = localStorage.getItem("selectedDataset");
    if (storedData) {
      try {
        const data = JSON.parse(storedData);
        setDatasetId(data.id);
        setDatasetName(data.name);
        console.log("Dataset ID:", data.id);
      } catch (error) {
        console.error("Error parsing dataset from localStorage:", error);
      }
    }
  }, []);

  // Connect to SSE when a preprocessing is started
  const connectToSse = useCallback(() => {
    if (eventSource) {
      eventSource.close();
    }

    console.log("Connecting to SSE at:", `${API_URL}api/sse-stream/`);

    // Create a new connection with the hardcoded URL
    const newEventSource = new EventSource(`${API_URL}api/sse-stream/`);
    setEventSource(newEventSource);

    // Set up event handlers
    newEventSource.onopen = () => {
      setSseConnected(true);
      setSseError(null);
      console.log("SSE connection established");
    };

    newEventSource.onerror = (error) => {
      console.error("SSE connection error:", error);
      setSseConnected(false);
      setSseError("Connection error. Will try to reconnect...");

      // Close the connection on error and reconnect after a delay
      newEventSource.close();
      setTimeout(connectToSse, 3000);
    };

    newEventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("SSE message received:", data);

        // Update the step status
        if (data.id && data.status) {
          setSteps((prevSteps) =>
            prevSteps.map((step) => {
              if (step.id === data.id) {
                return { ...step, status: data.status };
              }
              return step;
            })
          );

          // Update the current step index
          if (data.status === "processing") {
            setCurrentStepIndex(data.id - 1);
          }

          // Check if all steps are completed and we have a new dataset ID
          if (data.status === "completed" && data.cleaned_dataset_id) {
            // Store the new cleaned dataset ID
            const storedData = JSON.parse(
              localStorage.getItem("selectedDataset") || "{}"
            );

            // Create a new object with the cleaned dataset info
            const cleanedDataset = {
              id: data.cleaned_dataset_id,
              name: `${storedData.name || ""}_cleaned`,
            };

            // Update localStorage with the new dataset
            localStorage.setItem(
              "selectedDataset",
              JSON.stringify(cleanedDataset)
            );

            console.log(
              "Updated selected dataset to cleaned version:",
              cleanedDataset
            );

            // Update component state
            setDatasetId(data.cleaned_dataset_id);
            setDatasetName(`${storedData.name || ""}_cleaned`);

            // Set completion status
            setStatus("completed");

            // Set sample data from the backend if it exists
            // Update this part to ensure it's actually executing
            if (
              data.sample &&
              Array.isArray(data.sample) &&
              data.sample.length > 0
            ) {
              setSampleData(data.sample);
              console.log("Sample data received:", data.sample);
            } else {
              console.log("No sample data received, using fallback data");
              setSampleData([
                { id: 1, feature1: "Value 1", feature2: 42, target: 1 },
                { id: 2, feature1: "Value 2", feature2: 28, target: 0 },
                { id: 3, feature1: "Value 3", feature2: 35, target: 1 },
              ]);
            }
          }
        }
      } catch (error) {
        console.error("Error parsing SSE message:", error);
      }
    };

    return newEventSource;
  }, [eventSource, steps.length]);

  const handleDownload = async () => {
    try {
      // Show loading state if needed
      // setIsDownloading(true);

      // Make a request to the backend to get the CSV file
      const response = await fetch(
        `${API_URL}api/download-cleaned-dataset/${datasetId}/`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(
          `Failed to download: ${response.status} ${response.statusText}`
        );
      }

      // Get the blob from the response
      const blob = await response.blob();

      // Create a URL for the blob
      const url = window.URL.createObjectURL(blob);

      // Create a temporary anchor element
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;

      // Set the file name
      a.download = `${name || "dataset"}.csv`;

      // Add to the DOM and trigger the download
      document.body.appendChild(a);
      a.click();

      // Clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      // Reset loading state if needed
      // setIsDownloading(false);
    } catch (error) {
      console.error("Download failed:", error);
      // Handle error - maybe show a toast notification
      // setIsDownloading(false);
    }
  };

  // Cleanup function for SSE connection
  useEffect(() => {
    return () => {
      if (eventSource) {
        console.log("Closing SSE connection");
        eventSource.close();
      }
    };
  }, [eventSource]);

  const handlePreprocess = async () => {
    try {
      setStatus("processing");
      setCurrentStepIndex(-1);
      setSteps(INITIAL_STEPS);

      // Connect to SSE before starting preprocessing
      connectToSse();

      if (!datasetId) {
        console.error("No dataset selected. Please select a dataset first.");
        setStatus("error");
        return;
      }

      console.log(
        "Sending preprocessing request to:",
        `${API_URL}api/start-preprocessing/${datasetId}/`
      );

      // Call the Django API to start preprocessing with ID in URL path
      const response = await fetch(
        `${API_URL}api/start-preprocessing/${datasetId}/`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          // No need to include id in body since it's in the URL path
        }
      );

      // The status updates will come through SSE
      console.log("Preprocessing started successfully");
    } catch (error) {
      console.error("Preprocessing failed:", error);
      setStatus("error");
      if (eventSource) {
        eventSource.close();
      }
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
              {name || datasetName}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-4">
          <Button
            size="lg"
            onClick={handlePreprocess}
            disabled={status === "processing"}
            className="gap-2 bg-gradient-to-r from-primary to-chart-1 hover:opacity-90 transition-opacity relative group overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
            {status === "processing" ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Wand2 className="h-5 w-5" />
            )}
            <span className="relative z-10">Start Preprocessing</span>
          </Button>

          {status === "completed" && (
            <>
              <Button
                variant="outline"
                size="lg"
                onClick={handleDownload}
                className="gap-2 border-chart-2/30 hover:border-chart-2/50 transition-colors relative group overflow-hidden"
              >
                <div className="absolute inset-0 bg-chart-2/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                <Download className="h-5 w-5 text-chart-2" />
                <span className="relative z-10">Download Cleaned Dataset</span>
              </Button>

              <Button
                variant="secondary"
                size="lg"
                onClick={() => router.push("/visualization")}
                className="gap-2 bg-chart-3/20 hover:bg-chart-3/30 transition-colors relative group overflow-hidden"
              >
                <div className="absolute inset-0 bg-chart-3/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                <LineChart className="h-5 w-5 text-chart-3" />
                <span className="relative z-10">Visualize Data</span>
              </Button>
            </>
          )}
        </div>

        {/* Connection status indicator */}
        {status === "processing" && (
          <div className="mt-4 text-sm">
            {sseConnected ? (
              <p className="text-green-500 flex items-center gap-2">
                <span className="inline-block w-2 h-2 bg-green-500 rounded-full"></span>
                Connected to server
              </p>
            ) : (
              <p className="text-amber-500 flex items-center gap-2">
                <span className="inline-block w-2 h-2 bg-amber-500 rounded-full animate-pulse"></span>
                {sseError || "Connecting to server..."}
              </p>
            )}
          </div>
        )}
      </motion.div>

      <AnimatePresence>
        {status === "processing" && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-4 mb-8"
          >
            {steps.map((step, index) => (
              <ProcessingStep
                key={step.id}
                step={step}
                isActive={index === currentStepIndex}
                isCompleted={step.status === "completed"}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {status === "completed" && sampleData.length > 0 && (
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

            <div className="glass-effect rounded-xl border border-primary/20 relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-chart-1/5 via-transparent to-chart-2/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
              <div className="w-full overflow-x-auto">
                <DataTable data={sampleData} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}