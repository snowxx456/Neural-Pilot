import { DatasetType, ColumnType, ChartType, ColumnMetadata, ChartRecommendation } from './types'
import Papa from 'papaparse'

export async function analyzeCSV(file: File): Promise<DatasetType> {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        try {
          const columns = analyzeColumns(results.data)
          const recommendations = generateChartRecommendations(columns)
          
          resolve({
            data: results.data,
            columns,
            recommendations,
            filename: file.name,
            rowCount: results.data.length,
            columnCount: Object.keys(columns).length
          })
        } catch (error) {
          reject(error)
        }
      },
      error: (error) => {
        reject(error)
      }
    })
  })
}

function analyzeColumns(data: any[]): Record<string, ColumnMetadata> {
  const columns: Record<string, ColumnMetadata> = {}
  
  if (data.length === 0) return columns
  
  // Get column names from first row
  const columnNames = Object.keys(data[0])
  
  columnNames.forEach(name => {
    const values = data.map(row => row[name]).filter(val => val != null)
    const type = inferColumnType(values)
    
    columns[name] = {
      name,
      type,
      unique: new Set(values).size,
      missing: data.length - values.length,
      numeric: type === 'number',
      categorical: type === 'string' && new Set(values).size <= 20
    }
  })
  
  return columns
}

function inferColumnType(values: any[]): ColumnType {
  if (values.length === 0) return 'string'
  
  const sample = values.find(v => v !== null && v !== undefined)
  if (typeof sample === 'number') return 'number'
  if (typeof sample === 'boolean') return 'boolean'
  if (!isNaN(Date.parse(sample))) return 'date'
  return 'string'
}

function generateChartRecommendations(columns: Record<string, ColumnMetadata>): ChartRecommendation[] {
  const recommendations: ChartRecommendation[] = []
  const numericColumns = Object.entries(columns).filter(([_, col]) => col.numeric)
  const categoricalColumns = Object.entries(columns).filter(([_, col]) => col.categorical)
  
  // Bar Chart Recommendation
  if (categoricalColumns.length > 0 && numericColumns.length > 0) {
    recommendations.push({
      type: 'bar',
      title: `${numericColumns[0][1].name} by ${categoricalColumns[0][1].name}`,
      description: `Compare ${numericColumns[0][1].name} across different ${categoricalColumns[0][1].name} categories`,
      xAxis: categoricalColumns[0][0],
      yAxis: [numericColumns[0][0]],
      confidence: 90
    })
  }

  // Line Chart Recommendation
  if (numericColumns.length >= 2) {
    recommendations.push({
      type: 'line',
      title: `${numericColumns[0][1].name} vs ${numericColumns[1][1].name}`,
      description: `Analyze the relationship between ${numericColumns[0][1].name} and ${numericColumns[1][1].name}`,
      xAxis: numericColumns[0][0],
      yAxis: [numericColumns[1][0]],
      confidence: 85
    })
  }

  // Pie Chart Recommendation
  if (categoricalColumns.length > 0 && numericColumns.length > 0) {
    recommendations.push({
      type: 'pie',
      title: `Distribution of ${numericColumns[0][1].name} by ${categoricalColumns[0][1].name}`,
      description: `Show the proportion of ${numericColumns[0][1].name} across ${categoricalColumns[0][1].name} categories`,
      xAxis: categoricalColumns[0][0],
      yAxis: [numericColumns[0][0]],
      confidence: 80
    })
  }

  // Scatter Plot Recommendation
  if (numericColumns.length >= 2) {
    recommendations.push({
      type: 'scatter',
      title: `${numericColumns[0][1].name} vs ${numericColumns[1][1].name} Correlation`,
      description: `Explore the correlation between ${numericColumns[0][1].name} and ${numericColumns[1][1].name}`,
      xAxis: numericColumns[0][0],
      yAxis: [numericColumns[1][0]],
      confidence: 75
    })
  }

  return recommendations.sort((a, b) => b.confidence - a.confidence)
}