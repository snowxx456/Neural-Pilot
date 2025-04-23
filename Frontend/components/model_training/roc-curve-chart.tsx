"use client"

import { useEffect, useState } from "react"
import { Chart } from "@/components/ui/chart"
import { Line, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { fetchRocCurveData } from "@/lib/api"

interface RocCurvePoint {
  fpr: number
  tpr: number
}

interface RocCurveModel {
  name: string
  data: RocCurvePoint[]
  auc: number
}

interface RocCurveData {
  models: RocCurveModel[]
}

export function RocCurveChart() {
  const [data, setData] = useState<RocCurveData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const rocData = await fetchRocCurveData()
        setData(rocData)
      } catch (error) {
        console.error("Failed to fetch ROC curve data:", error)
        // Provide fallback data
        setData({
          models: [
            {
              name: "RandomForest",
              data: Array.from({ length: 100 }, (_, i) => {
                const x = i / 100
                const y = Math.min(1, x + 0.4 * Math.sin(Math.PI * x))
                return { fpr: x, tpr: y }
              }),
              auc: 0.92,
            },
            {
              name: "LogisticRegression",
              data: Array.from({ length: 100 }, (_, i) => {
                const x = i / 100
                const y = Math.min(1, x + 0.25 * Math.sin(Math.PI * x))
                return { fpr: x, tpr: y }
              }),
              auc: 0.82,
            },
          ],
        })
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  if (loading) {
    return <div className="flex justify-center items-center h-[400px]">Loading ROC curve data...</div>
  }

  if (!data) {
    return <div className="flex justify-center items-center h-[400px]">No ROC curve data available</div>
  }

  // Transform data for Recharts
  const chartData = data.models[0].data.map((point, index) => {
    const dataPoint: any = {
      fpr: point.fpr,
    }

    // Add TPR values for each model
    data.models.forEach((model) => {
      dataPoint[model.name] = model.data[index].tpr
    })

    return dataPoint
  })

  // Add the random baseline
  chartData.forEach((point) => {
    point.baseline = point.fpr
  })

  // Generate colors for each model
  const getModelColor = (index: number) => {
    const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088fe"]
    return colors[index % colors.length]
  }

  return (
    <div className="w-full h-[400px]">
      <Chart>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              dataKey="fpr"
              label={{ value: "False Positive Rate", position: "insideBottom", offset: -10 }}
              domain={[0, 1]}
            />
            <YAxis label={{ value: "True Positive Rate", angle: -90, position: "insideLeft" }} domain={[0, 1]} />
            <Tooltip
              formatter={(value: number) => [value.toFixed(3), "TPR"]}
              labelFormatter={(value) => `FPR: ${Number(value).toFixed(3)}`}
            />
            <Legend />

            {/* Random baseline */}
            <Line
              type="monotone"
              dataKey="baseline"
              stroke="#777777"
              strokeDasharray="5 5"
              name="Random Baseline"
              dot={false}
            />

            {/* Model curves */}
            {data.models.map((model, index) => (
              <Line
                key={model.name}
                type="monotone"
                dataKey={model.name}
                stroke={getModelColor(index)}
                name={`${model.name} (AUC = ${model.auc.toFixed(2)})`}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Chart>
    </div>
  )
}
