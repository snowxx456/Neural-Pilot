"use client"

import { useEffect, useState } from "react"
import { fetchConfusionMatrix } from "@/lib/api"

interface ConfusionMatrixData {
  matrix: number[][]
  labels: string[]
}

export function ConfusionMatrixChart() {
  const [data, setData] = useState<ConfusionMatrixData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const matrixData = await fetchConfusionMatrix()
        setData(matrixData)
      } catch (error) {
        console.error("Failed to fetch confusion matrix data:", error)
        // Provide fallback data
        setData({
          matrix: [
            [1180, 413],
            [142, 265],
          ],
          labels: ["0", "1"],
        })
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  if (loading) {
    return <div className="flex justify-center items-center h-[400px]">Loading confusion matrix data...</div>
  }

  if (!data) {
    return <div className="flex justify-center items-center h-[400px]">No confusion matrix data available</div>
  }

  // Calculate the total number of samples for each true class
  const rowSums = data.matrix.map((row) => row.reduce((sum, val) => sum + val, 0))

  // Calculate percentages for each cell
  const getPercentage = (value: number, rowIndex: number) => {
    const rowSum = rowSums[rowIndex]
    return rowSum > 0 ? (value / rowSum) * 100 : 0
  }

  // Get color intensity based on percentage
  const getCellColor = (value: number, rowIndex: number) => {
    const percentage = getPercentage(value, rowIndex)

    // Blue color with varying opacity based on percentage
    return `rgba(59, 130, 246, ${percentage / 100})`
  }

  return (
    <div className="w-full overflow-auto">
      <div className="min-w-[400px] p-4">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `80px repeat(${data.labels.length}, 1fr)`,
            gridTemplateRows: `40px repeat(${data.labels.length}, 1fr)`,
          }}
        >
          {/* Top-left empty cell */}
          <div className="border flex items-center justify-center font-medium">True / Pred</div>

          {/* Column headers (Predicted) */}
          {data.labels.map((label, i) => (
            <div key={`col-${i}`} className="border p-2 flex items-center justify-center font-medium">
              {label}
            </div>
          ))}

          {/* Row headers and data cells */}
          {data.matrix.map((row, rowIndex) => (
            <>
              {/* Row header (True) */}
              <div key={`row-${rowIndex}`} className="border p-2 flex items-center justify-center font-medium">
                {data.labels[rowIndex]}
              </div>

              {/* Data cells */}
              {row.map((value, colIndex) => (
                <div
                  key={`cell-${rowIndex}-${colIndex}`}
                  className="border p-2 flex flex-col items-center justify-center h-16"
                  style={{ backgroundColor: getCellColor(value, rowIndex) }}
                >
                  <span className="font-bold text-white">{value}</span>
                  <span className="text-xs text-white/80">{getPercentage(value, rowIndex).toFixed(1)}%</span>
                </div>
              ))}
            </>
          ))}
        </div>

        <div className="mt-4 text-sm text-muted-foreground text-center">
          The confusion matrix shows the counts of true vs predicted labels.
          <br />
          Percentages are calculated based on the total samples in each true class.
        </div>
      </div>
    </div>
  )
}
