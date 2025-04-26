"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Clock, Info, Award, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface ModelMetrics {
  precision?: number;
  recall?: number;
  f1?: number;
  [key: string]: number | undefined;
}

interface CVScore {
  mean: number;
  std: number;
}

interface ModelResult {
  name: string;
  accuracy: number;
  metrics: ModelMetrics;
  report: string;
  trainingTime: number;
  isBest: boolean;
  description?: string;
  parameters: any;
  cvScore: CVScore;
  cvTime: number;
}

interface ModelCardProps {
  model: ModelResult;
}

export function ModelCard({ model }: ModelCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showSparkles, setShowSparkles] = useState(false);
  const [scale, setScale] = useState(1);

  // Animation effects for best model
  useEffect(() => {
    if (model.isBest) {
      // Pulsing effect
      const pulseInterval = setInterval(() => {
        setScale((prev) => (prev === 1 ? 1.03 : 1));
      }, 2000);

      // Sparkle effect
      const sparkleInterval = setInterval(() => {
        setShowSparkles(true);
        setTimeout(() => setShowSparkles(false), 1000);
      }, 3000);

      return () => {
        clearInterval(pulseInterval);
        clearInterval(sparkleInterval);
      };
    }
  }, [model.isBest]);

  // Determine badge color based on accuracy
  const getBadgeVariant = (accuracy: number) => {
    if (accuracy >= 0.8) return "success";
    if (accuracy >= 0.7) return "default";
    if (accuracy >= 0.6) return "warning";
    return "destructive";
  };

  // Format time in seconds to be more readable
  const formatTime = (seconds: number) => {
    if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
    if (seconds < 60) return `${seconds.toFixed(2)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  };

  // Extract metrics from the report string if metrics object is empty
  const extractMetrics = () => {
    const metricsFromReport = {
      precision: 0,
      recall: 0,
      f1: 0,
    };

    // For regression models, we'll use R2 Score as precision, RMSE as recall, and MAE as f1
    // (These aren't accurate translations but prevent the component from breaking)
    if (model.report) {
      const reportLines = model.report.split("\n");

      reportLines.forEach((line) => {
        if (line.includes("R2 Score:")) {
          metricsFromReport.precision = parseFloat(line.split(":")[1].trim());
        } else if (line.includes("Root Mean Squared Error:")) {
          // Convert RMSE to a 0-1 scale roughly
          const rmse = parseFloat(line.split(":")[1].trim());
          metricsFromReport.recall = Math.max(0, 1 - rmse / 10); // Assuming RMSE < 10 is good
        } else if (line.includes("Mean Absolute Error:")) {
          // Convert MAE to a 0-1 scale roughly
          const mae = parseFloat(line.split(":")[1].trim());
          metricsFromReport.f1 = Math.max(0, 1 - mae / 5); // Assuming MAE < 5 is good
        }
      });
    }

    return metricsFromReport;
  };

  // Get the metrics either from the model.metrics object or extract from report
  const metrics =
    Object.keys(model.metrics || {}).length === 0
      ? extractMetrics()
      : model.metrics;

  return (
    <Card
      className={cn(
        "overflow-hidden transition-all duration-500 relative",
        model.isBest
          ? "border-2 border-primary shadow-lg"
          : "border-transparent"
      )}
      style={{ transform: model.isBest ? `scale(${scale})` : "scale(1)" }}
    >
      {model.isBest && (
        <div className="absolute -top-3 -right-3 z-10">
          <div className="relative">
            <Award
              className="h-12 w-12 text-yellow-400 drop-shadow-md"
              fill="currentColor"
            />
            {showSparkles && (
              <>
                <Sparkles className="absolute -top-2 -left-2 h-4 w-4 text-yellow-300 animate-pulse" />
                <Sparkles className="absolute -bottom-2 -right-2 h-4 w-4 text-yellow-300 animate-pulse" />
              </>
            )}
          </div>
        </div>
      )}

      {model.isBest && (
        <div className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-primary/10 animate-gradient-x"></div>
        </div>
      )}

      <CardHeader className={cn("pb-2", model.isBest && "bg-primary/5")}>
        <div className="flex justify-between items-start">
          <div>
            <div>
              <CardTitle className="text-xl">{model.name}</CardTitle>
              <div className="flex items-center mt-1">
                {model.isBest && (
                  <Badge className="mr-2" variant="outline">
                    Best Model
                  </Badge>
                )}
                <CardDescription>
                  {model.description || "Regression model"}
                </CardDescription>
              </div>
            </div>
          </div>
          <Badge variant={getBadgeVariant(model.accuracy) as any}>
            {(model.accuracy * 100).toFixed(2)}%
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>R² Score</span>
              <span className="font-medium">{model.accuracy.toFixed(4)}</span>
            </div>
            <div className="mt-2">
              <Progress
                value={model.accuracy * 100}
                className={cn(
                  "h-2",
                  model.isBest && "bg-primary/20",
                  model.isBest ? "[&>div]:animate-pulse" : ""
                )}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Accuracy</p>
              <p className="font-medium">
                {metrics.precision?.toFixed(4) || model.accuracy.toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground">Error</p>
              <p className="font-medium">
                {metrics.recall?.toFixed(4) || (1 - model.accuracy).toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground">CV Score</p>
              <p className="font-medium">
                {model.cvScore?.mean.toFixed(4) || "N/A"}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground">Training Time</p>
              <p className="font-medium flex items-center">
                <Clock className="h-3 w-3 mr-1 inline" />
                {formatTime(model.trainingTime)}
              </p>
            </div>
          </div>

          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" className="w-full">
                <Info className="h-4 w-4 mr-2" />
                Detailed Report
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-3xl">
              <DialogHeader>
                <DialogTitle>{model.name} - Performance Report</DialogTitle>
                <DialogDescription>
                  Detailed metrics and evaluation results
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 mt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">Regression Report</h4>
                    <pre className="bg-muted p-4 rounded-md text-xs overflow-auto max-h-[300px]">
                      {model.report || "No report available"}
                    </pre>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">
                      Cross-Validation Results
                    </h4>
                    <div className="bg-muted p-4 rounded-md">
                      <p className="text-sm">
                        Mean CV Score: {model.cvScore?.mean.toFixed(4) || "N/A"}
                      </p>
                      <p className="text-sm">
                        Standard Deviation:{" "}
                        {model.cvScore?.std.toFixed(4) || "N/A"}
                      </p>
                      <p className="text-sm">
                        CV Time: {formatTime(model.cvTime || 0)}
                      </p>
                    </div>

                    <h4 className="font-medium mt-4 mb-2">Model Parameters</h4>
                    <div className="bg-muted p-4 rounded-md text-xs overflow-auto max-h-[150px]">
                      <pre>
                        {JSON.stringify(model.parameters || {}, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </CardContent>
    </Card>
  );
}
