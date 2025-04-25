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
import { DatasetType, ChartConfig } from "@/lib/types";
import { ChartDisplay } from "@/components/visualization/ChartDisplay";
import { ChartControls } from "@/components/visualization/ChartControls";
import { DataSummary } from "@/components/visualization/DataSummary";
import { RotateCcw } from "lucide-react";

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

  useEffect(() => {
    if (dataset.recommendations?.length > 0) {
      const recommendation = dataset.recommendations[0];
      setChartConfig({
        type: recommendation.type,
        title: recommendation.title,
        xAxisColumn: recommendation.xAxis || "",
        yAxisColumns: recommendation.yAxis || [],
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
      });
    }
  }, [dataset]);

  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <Button
          variant="outline"
          onClick={onReset}
          className="flex items-center gap-2 glass neon-border"
        >
          <RotateCcw className="h-4 w-4" />
          Upload New File
        </Button>
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
                    <CardTitle className="gradient-text">
                      {chartConfig.title}
                    </CardTitle>
                    <CardDescription className="opacity-70">
                      {dataset.recommendations[0]?.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="overflow-x-auto">
                    <div className="h-[600px] w-full">
                      <ChartDisplay data={dataset.data} config={chartConfig} />
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="data" className="m-0">
          <DataSummary dataset={dataset} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
