"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { Dataset } from "@/lib/types";
import { Download, ExternalLink, FileText, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import axios from "axios";
import { useToast } from "@/hooks/use-toast";

interface DatasetCardProps {
  dataset: Dataset;
  onClick: () => void;
}
const API = process.env.NEXT_PUBLIC_SERVER_URL || "http://localhost:8000/";

export function DatasetCard({ dataset, onClick }: DatasetCardProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [isSelecting, setIsSelecting] = useState(false);
  const { toast } = useToast();

  const handleDownload = async (
    e: React.MouseEvent<HTMLButtonElement>,
    dataset: Dataset
  ) => {
    e.stopPropagation();
    setIsLoading(true);

    try {
      console.log("Attempting to download dataset:", dataset.id);

      const response = await axios.post(
        `${API}api/dataset/download/`,
        {
          datasetRef: dataset.id,
        },
        {
          responseType: "blob",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.status === 200 && response.data) {
        const blob = new Blob([response.data], {
          type: response.headers["content-type"] || "text/csv",
        });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute(
          "download",
          `${dataset.title.replace(/[^a-zA-Z0-9]/g, "_")}.csv`
        );
        document.body.appendChild(link);
        link.click();
        window.URL.revokeObjectURL(url);
        link.remove();

        toast({
          title: "Success",
          description: "Dataset downloaded successfully",
        });
      }
    } catch (error: any) {
      console.error("Download error:", error);
      toast({
        title: "Download Failed",
        description: error.message || "Failed to download dataset",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelect = async (
    e: React.MouseEvent<HTMLButtonElement>,
    dataset: Dataset
  ) => {
    e.stopPropagation();
    setIsSelecting(true);

    try {
      console.log("Selecting dataset:", dataset.id);

      // Fetch the dataset from Kaggle and store it in the backend
      const response = await axios.post(
        `${API}api/dataset/select/`,
        {
          datasetRef: dataset.id,
          datasetUrl: dataset.externalLink,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.status === 200) {
        // Create the dataset info object
        const datasetInfo = {
          id: dataset.id,
          name: dataset.title,
          timestamp: new Date().toISOString(),
        };

        // Replace any existing dataset in local storage
        localStorage.setItem("selectedDataset", JSON.stringify(datasetInfo));

        toast({
          title: "Dataset Selected",
          description:
            "Dataset has been successfully imported to your workspace",
        });
      }
    } catch (error: any) {
      console.error("Selection error:", error);
      toast({
        title: "Selection Failed",
        description: error.message || "Failed to select dataset",
        variant: "destructive",
      });
    } finally {
      setIsSelecting(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className={cn(
        "relative flex flex-col h-full overflow-hidden",
        "glass-effect rounded-xl",
        "transition-all hover:border-primary/20 cursor-pointer",
        "group"
      )}
      onClick={onClick}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-chart-1/5 via-transparent to-chart-2/5 opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-chart-1 to-chart-2 transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />

      <div className="p-6 relative">
        <div className="mb-4 flex items-center justify-between">
          <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center group-hover:scale-110 transition-transform">
            <Sparkles className="h-6 w-6 text-primary animate-pulse-slow" />
          </div>

          <div className="px-3 py-1.5 rounded-full glass-effect text-chart-2 text-xs font-medium border border-chart-2/20">
            {dataset.category || "Dataset"}
          </div>
        </div>

        <h3 className="text-lg font-medium mb-2 line-clamp-1 group-hover:text-glow transition-all">
          {dataset.title}
        </h3>

        <p className="text-sm text-muted-foreground mb-6 line-clamp-3">
          {dataset.description}
        </p>

        <div className="mt-auto flex gap-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 gap-1.5 group/btn"
            onClick={(e) => handleSelect(e, dataset)}
            disabled={isSelecting}
          >
            {isSelecting ? (
              <>
                <span className="animate-spin">⏳</span>
                <span>Selecting...</span>
              </>
            ) : (
              <>
                <ExternalLink className="h-4 w-4 group-hover/btn:text-chart-1 transition-colors" />
                <span>Select</span>
              </>
            )}
          </Button>

          <Button
            onClick={(e) => handleDownload(e, dataset)}
            disabled={isLoading}
            variant="default"
            size="sm"
            className="flex-1 gap-1.5"
          >
            {isLoading ? (
              <>
                <span className="animate-spin">⏳</span>
                <span>Downloading...</span>
              </>
            ) : (
              <>
                <Download className="h-4 w-4" />
                <span>Download</span>
              </>
            )}
          </Button>
        </div>
      </div>
    </motion.div>
  );
}
