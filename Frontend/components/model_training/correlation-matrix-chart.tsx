"use client"

import { useEffect, useState } from "react"
import { fetchCorrelationMatrix } from "@/lib/api"

interface CorrelationData {
  features: string[]
  matrix: number[][]
}

export function CorrelationMatrixChart() {
  const [data, setData] = useState<CorrelationData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const correlationData = await fetchCorrelationMatrix()
        setData(correlationData)
      } catch (error) {
        console.error("Failed to fetch correlation matrix data:", error)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  if (loading) {
    return <div className="flex justify-center items-center h-[400px]">Loading correlation matrix data...</div>
  }

  if (!data) {
    return <div className="flex justify-center items-center h-[400px]">No correlation matrix data available</div>
  }

  // Get color based on correlation value
  const getColor = (value: number) => {
    // Blue for positive correlations, red for negative
    if (value >= 0) {
      return `rgba(59, 130, 246, ${Math.abs(value)})`
    } else {
      return `rgba(239, 68, 68, ${Math.abs(value)})`
    }
  }

  return (
    <div className="w-full overflow-auto">
      <div className="min-w-[600px] p-4">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `80px repeat(${data.features.length}, 1fr)`,
            gridTemplateRows: `40px repeat(${data.features.length}, 1fr)`,
          }}
        >
          {/* Top-left empty cell */}
          <div className="border flex items-center justify-center font-medium">Features</div>

          {/* Column headers */}
          {data.features.map((feature, i) => (
            <div key={`col-${i}`} className="border p-2 flex items-center justify-center font-medium text-xs">
              {feature}
            </div>
          ))}

          {/* Row headers and data cells */}
          {data.matrix.map((row, rowIndex) => (
            <>
              {/* Row header */}
              <div key={`row-${rowIndex}`} className="border p-2 flex items-center justify-center font-medium text-xs">
                {data.features[rowIndex]}
              </div>

              {/* Data cells */}
              {row.map((value, colIndex) => (
                <div
                  key={`cell-${rowIndex}-${colIndex}`}
                  className="border p-2 flex items-center justify-center h-12"
                  style={{ backgroundColor: getColor(value) }}
                >
                  <span className={`font-bold text-xs ${Math.abs(value) > 0.5 ? "text-white" : "text-black"}`}>
                    {value.toFixed(2)}
                  </span>
                </div>
              ))}
            </>
          ))}
        </div>

        <div className="mt-4 text-sm text-muted-foreground text-center">
          The correlation matrix shows the relationship between numerical features.
          <br />
          Values range from -1 (strong negative correlation) to 1 (strong positive correlation).
        </div>
      </div>
    </div>
  )
}
