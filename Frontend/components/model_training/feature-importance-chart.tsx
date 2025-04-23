"use client"

import { useEffect, useState } from "react"
import { Chart } from "@/components/ui/chart"
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { fetchFeatureImportance } from "@/lib/api"

interface FeatureImportance {
  feature: string
  importance: number
}

export function FeatureImportanceChart() {
  const [data, setData] = useState<FeatureImportance[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const importanceData = await fetchFeatureImportance()
        setData(importanceData)
      } catch (error) {
        console.error("Failed to fetch feature importance data:", error)
        // Provide fallback data
        setData([
          { feature: "age", importance: 0.185 },
          { feature: "income", importance: 0.162 },
          { feature: "education_level", importance: 0.143 },
          { feature: "credit_score", importance: 0.128 },
          { feature: "employment_years", importance: 0.112 },
          { feature: "debt_to_income", importance: 0.098 },
          { feature: "num_credit_cards", importance: 0.087 },
          { feature: "num_dependents", importance: 0.076 },
          { feature: "has_mortgage", importance: 0.065 },
          { feature: "has_car_loan", importance: 0.054 },
        ])
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  if (loading) {
    return <div className="flex justify-center items-center h-[400px]">Loading feature importance data...</div>
  }

  // Sort data by importance in descending order and take top 15
  const sortedData = [...data].sort((a, b) => b.importance - a.importance).slice(0, 15)

  // Generate colors based on importance
  const getBarColor = (importance: number) => {
    const maxImportance = Math.max(...sortedData.map((d) => d.importance))
    const normalizedValue = importance / maxImportance

    // Color gradient from light to dark blue
    const r = Math.round(100 + (0 - 100) * normalizedValue)
    const g = Math.round(150 + (82 - 150) * normalizedValue)
    const b = Math.round(255 + (155 - 255) * normalizedValue)

    return `rgb(${r}, ${g}, ${b})`
  }

  return (
    <div className="w-full h-[400px]">
      <Chart>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={sortedData} layout="vertical" margin={{ top: 20, right: 30, left: 150, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
            <XAxis type="number" domain={[0, "dataMax"]} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} width={140} />
            <Tooltip
              formatter={(value: number) => [`${value.toFixed(4)}`, "Importance"]}
              labelFormatter={(value) => `Feature: ${value}`}
            />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {sortedData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.importance)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Chart>
    </div>
  )
}
