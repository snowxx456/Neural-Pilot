"use client"
import type { ModelResult } from "@/lib/types"
import { Chart } from "@/components/ui/chart"
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { TooltipProps } from "recharts"

interface ModelComparisonChartProps {
  models: ModelResult[]
}

export function ModelComparisonChart({ models }: ModelComparisonChartProps) {
  // Transform the model data for the chart
  const chartData = models.map((model) => ({
    name: model.name,
    accuracy: Number.parseFloat((model.accuracy * 100).toFixed(2)),
    precision: Number.parseFloat((model.metrics.precision * 100).toFixed(2)),
    recall: Number.parseFloat((model.metrics.recall * 100).toFixed(2)),
    f1: Number.parseFloat((model.metrics.f1 * 100).toFixed(2)),
    trainingTime: model.trainingTime,
    isBest: model.isBest,
  }))

  return (
    <div className="w-full aspect-[16/9]">
      <Chart>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 70 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 12 }} />
            <YAxis yAxisId="left" label={{ value: "Percentage (%)", angle: -90, position: "insideLeft" }} />
            <YAxis
              yAxisId="right"
              orientation="right"
              label={{ value: "Training Time (s)", angle: 90, position: "insideRight" }}
            />
            <Tooltip
              formatter={(value: number | string, name: string) => {
                if (name === "trainingTime") {
                  return [typeof value === 'number' ? `${value.toFixed(2)}s` : value, "Training Time"];
                }
                return [`${value}%`, name.charAt(0).toUpperCase() + name.slice(1)];
              }}
            />
            <Legend />
            <Bar yAxisId="left" dataKey="accuracy" fill="#8884d8" name="Accuracy" radius={[4, 4, 0, 0]} />
            <Bar yAxisId="left" dataKey="precision" fill="#82ca9d" name="Precision" radius={[4, 4, 0, 0]} />
            <Bar yAxisId="left" dataKey="recall" fill="#ffc658" name="Recall" radius={[4, 4, 0, 0]} />
            <Bar yAxisId="left" dataKey="f1" fill="#ff8042" name="F1 Score" radius={[4, 4, 0, 0]} />
            <Bar
              yAxisId="right"
              dataKey="trainingTime"
              fill="#ff0000"
              name="Training Time"
              radius={[4, 4, 0, 0]}
              opacity={0.6}
            />
          </BarChart>
        </ResponsiveContainer>
      </Chart>
    </div>
  )
}
