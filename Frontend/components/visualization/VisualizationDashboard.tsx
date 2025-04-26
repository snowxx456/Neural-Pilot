"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { DatasetType, ChartConfig, ChartType } from "@/lib/types";
import { ChartDisplay } from "@/components/visualization/ChartDisplay";
import { ChartControls } from "@/components/visualization/ChartControls";
import { DataSummary } from "@/components/visualization/DataSummary";
import { RotateCcw, ArrowRight, ChevronRight } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

interface VisualizationDashboardProps {
  dataset: DatasetType;
  onReset: () => void;
}

export function VisualizationDashboard({
  dataset,
  onReset,
}: VisualizationDashboardProps) {
  const [activeTab, setActiveTab] = useState("visualize");
  const [chartConfig, setChartConfig] = useState<ChartConfig | null>(null);
  const [activeRecommendation, setActiveRecommendation] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize with the first recommendation when dataset loads
  useEffect(() => {
    if (dataset?.recommendations?.length > 0) {
      initializeChart(0);
      setIsLoading(false);
    } else {
      // If no recommendations, create a default chart based on data types
      createDefaultChart();
      setIsLoading(false);
    }
  }, [dataset]);

  // Initialize chart based on recommendation index
  const initializeChart = (index: number) => {
    if (!dataset?.recommendations?.[index]) return;

    const recommendation = dataset.recommendations[index];
    setActiveRecommendation(index);

    const chartType = determineChartType(
      recommendation.type,
      recommendation.xAxis,
      recommendation.yAxis
    );

    setChartConfig({
      type: chartType,
      title:
        recommendation.title ||
        `${recommendation.xAxis} vs ${recommendation.yAxis?.join(", ")}`,
      xAxisColumn: recommendation.xAxis || "",
      yAxisColumns: recommendation.yAxis || [],
      colors: [
        "hsl(340 82% 52%)", // Red
        "hsl(262 83% 58%)", // Purple
        "hsl(190 95% 39%)", // Blue
        "hsl(130 94% 39%)", // Green
        "hsl(45 93% 47%)", // Yellow
      ],
      showLegend: true,
      showGrid: true,
      showTooltip: true,
      animation: true,
      stacked: false,
      filter: recommendation.filter || {},
    });
  };

  // Determine appropriate chart type based on data characteristics
  const determineChartType = (
    suggestedType: ChartType | undefined,
    xAxis: string | undefined,
    yAxis: string[] | undefined
  ): ChartType => {
    // If a valid type is suggested, use that
    if (
      suggestedType &&
      ["bar", "line", "scatter", "pie", "area", "histogram"].includes(
        suggestedType
      )
    ) {
      return suggestedType as ChartType;
    }

    // Get column metadata
    const xColumnMeta = xAxis ? dataset.columns[xAxis] : undefined;

    // If no X axis specified, default to bar chart
    if (!xAxis) return "bar";

    // If no Y axis but X axis is categorical, use histogram
    if (!yAxis || yAxis.length === 0) {
      return "histogram";
    }

    // Recommend chart type based on data characteristics
    if (xColumnMeta?.type === "datetime" || xColumnMeta?.type === "date") {
      return "line"; // Time series data
    } else if (
      xColumnMeta?.type === "categorical" ||
      xColumnMeta?.type === "string"
    ) {
      if (yAxis.length > 1) {
        return "bar"; // Multiple categories
      }
      // Check uniqueness for categorical data
      const uniqueValues = new Set(dataset.data.map((item) => item[xAxis]));
      if (uniqueValues.size <= 8) {
        return "pie"; // Small number of categories
      }
      return "bar"; // Many categories
    } else if (
      xColumnMeta?.type === "numeric" ||
      xColumnMeta?.type === "number"
    ) {
      if (yAxis.length > 1) {
        return "area"; // Multiple numeric series
      }
      return "scatter"; // Numeric correlation
    }

    // Default fallback
    return "bar";
  };

  // Create a default chart when no recommendations are available
  const createDefaultChart = () => {
    // Find first numeric and first categorical/string columns
    const columns = Object.entries(dataset.columns);
    const numericColumn = columns.find(
      ([_, meta]) => meta.type === "numeric" || meta.type === "number"
    );
    const categoricalColumn = columns.find(
      ([_, meta]) => meta.type === "categorical" || meta.type === "string"
    );

    // Default to first two columns if specific types not found
    const xColumn = categoricalColumn?.[0] || columns[0]?.[0] || "";
    const yColumn =
      numericColumn?.[0] ||
      (columns[1]?.[0] !== xColumn ? columns[1]?.[0] : columns[0]?.[0]) ||
      "";

    const chartType = categoricalColumn ? "bar" : "line";

    setChartConfig({
      type: chartType,
      title: `${xColumn} vs ${yColumn}`,
      xAxisColumn: xColumn,
      yAxisColumns: [yColumn],
      colors: [
        "hsl(340 82% 52%)",
        "hsl(262 83% 58%)",
        "hsl(190 95% 39%)",
        "hsl(130 94% 39%)",
        "hsl(45 93% 47%)",
      ],
      showLegend: true,
      showGrid: true,
      showTooltip: true,
      animation: true,
      stacked: false,
      filter: {},
    });
  };

  // Handle recommendation navigation
  const navigateRecommendation = (direction: "next" | "prev") => {
    if (!dataset.recommendations || dataset.recommendations.length <= 1) return;

    let newIndex = activeRecommendation;
    if (direction === "next") {
      newIndex = (activeRecommendation + 1) % dataset.recommendations.length;
    } else {
      newIndex =
        (activeRecommendation - 1 + dataset.recommendations.length) %
        dataset.recommendations.length;
    }

    initializeChart(newIndex);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold gradient-text">Data Visualization</h2>
      </div>

      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="space-y-6"
      >
        <TabsList className="grid w-[400px] grid-cols-2 glass">
          <TabsTrigger
            value="visualize"
            className="data-[state=active]:neon-border"
          >
            Visualize
          </TabsTrigger>
          <TabsTrigger value="data" className="data-[state=active]:neon-border">
            Data Summary
          </TabsTrigger>
        </TabsList>

        <TabsContent value="visualize" className="m-0">
          {isLoading ? (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="lg:col-span-1">
                <Skeleton className="h-[800px] w-full" />
              </div>
              <div className="lg:col-span-3">
                <Skeleton className="h-[800px] w-full" />
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="lg:col-span-1">
                {chartConfig && (
                  <ChartControls
                    config={chartConfig}
                    setConfig={setChartConfig}
                    columns={dataset.columns}
                  />
                )}
              </div>

              <div className="lg:col-span-3">
                {chartConfig && (
                  <Card className="h-full glass neon-border glow chart-container">
                    <CardHeader>
                      <div className="flex justify-between items-center">
                        <div>
                          <CardTitle className="gradient-text">
                            {chartConfig.title}
                          </CardTitle>
                          {dataset.recommendations &&
                            dataset.recommendations.length > 0 && (
                              <CardDescription className="opacity-70">
                                {dataset.recommendations[activeRecommendation]
                                  ?.description ||
                                  "Visualize your data insights"}
                              </CardDescription>
                            )}
                        </div>

                        {dataset.recommendations &&
                          dataset.recommendations.length > 1 && (
                            <div className="flex items-center gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => navigateRecommendation("prev")}
                                className="h-8 w-8 p-0 rounded-full"
                              >
                                <ChevronRight className="h-4 w-4 rotate-180" />
                              </Button>
                              <span className="text-xs">
                                {activeRecommendation + 1} /{" "}
                                {dataset.recommendations.length}
                              </span>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => navigateRecommendation("next")}
                                className="h-8 w-8 p-0 rounded-full"
                              >
                                <ChevronRight className="h-4 w-4" />
                              </Button>
                            </div>
                          )}
                      </div>
                    </CardHeader>
                    <CardContent className="overflow-x-auto">
                      <div className="h-[600px] w-full">
                        <ChartDisplay
                          data={dataset.data}
                          config={chartConfig}
                        />
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="data" className="m-0">
          <DataSummary dataset={dataset} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
