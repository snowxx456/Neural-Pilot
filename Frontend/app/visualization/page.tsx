"use client";

import { useState } from "react";
import { FileUploader } from "@/components/visualization/FileUploader";
import { VisualizationDashboard } from "@/components/visualization/VisualizationDashboard";
import { DatasetType } from "@/lib/types";
import { DataHeader } from "@/components/visualization/DataHeader";
import { useEffect } from "react";

export default function Home() {
  const [dataset, setDataset] = useState<DatasetType | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState<any>(null);
  const [datasetName, setDatasetName] = useState<string>("");
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

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <DataHeader dataset={dataset} />

        {!dataset ? (
          <div className="flex items-center justify-center h-[70vh]">
            <FileUploader
              datasetId={datasetId} // Replace with appropriate logic to generate or fetch datasetId
              onDatasetReady={setDataset}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
              error={error}
              setError={setError}
              datasetname={datasetName} // Pass the dataset name to the FileUploader
            />
          </div>
        ) : (
          <VisualizationDashboard
            dataset={dataset}
            onReset={() => setDataset(null)}
          />
        )}
      </div>
    </main>
  );
}
