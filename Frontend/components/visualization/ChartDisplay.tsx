'use client'

import { useEffect, useState } from 'react'
import { 
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter, 
  PieChart, Pie, AreaChart, Area, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Label
} from 'recharts'
import { ChartConfig } from '@/lib/types'
import { Skeleton } from '@/components/ui/skeleton'

interface ChartDisplayProps {
  data: Record<string, any>[];
  config: ChartConfig;
}

export function ChartDisplay({ data, config }: ChartDisplayProps) {
  const [processedData, setProcessedData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    setLoading(true)
    
    const prepareData = () => {
      try {
        let result = [...data]
        
        if (config.filter) {
          Object.entries(config.filter).forEach(([key, value]) => {
            result = result.filter(item => item[key] === value)
          })
        }
        
        if (config.type === 'pie' || config.type === 'donut') {
          const counts: Record<string, number> = {}
          data.forEach(item => {
            const value = String(item[config.xAxisColumn])
            counts[value] = (counts[value] || 0) + 1
          })
          
          result = Object.entries(counts).map(([name, value]) => ({
            name,
            value
          }))
        }
        
        if (result.length > 1000) {
          result = result.slice(0, 1000)
        }
        
        setProcessedData(result)
      } catch (err) {
        console.error('Error processing chart data:', err)
        setProcessedData([])
      } finally {
        setLoading(false)
      }
    }
    
    prepareData()
  }, [data, config])
  
  if (loading) {
    return <Skeleton className="h-full w-full" />
  }
  
  if (!processedData || processedData.length === 0) {
    return <div className="flex items-center justify-center h-full">No data available for this chart type</div>
  }
  
  const axisStyle = {
    fontSize: '12px',
    fontWeight: 500,
    fill: 'currentColor',
  }

  const labelStyle = {
    fontSize: '14px',
    fontWeight: 600,
    fill: 'currentColor',
  }
  
  const renderChart = () => {
    const colors = config.colors
    
    switch (config.type) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 40, bottom: 70 }}
            >
              {config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />}
              <XAxis 
                dataKey={config.xAxisColumn} 
                angle={-45} 
                textAnchor="end"
                interval={0}
                height={70}
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.xAxisColumn} 
                  position="bottom" 
                  offset={50}
                  style={labelStyle}
                />
              </XAxis>
              <YAxis 
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.yAxisColumns[0]} 
                  position="left" 
                  angle={-90} 
                  offset={-20}
                  style={labelStyle}
                />
              </YAxis>
              {config.showTooltip && <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: 'none' }} />}
              {config.showLegend && <Legend />}
              {config.yAxisColumns.map((column, index) => (
                <Bar 
                  key={column} 
                  dataKey={column} 
                  fill={colors[index % colors.length]} 
                  stackId={config.stacked ? "stack" : undefined}
                  isAnimationActive={config.animation}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )
        
      case 'line':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 40, bottom: 70 }}
            >
              {config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />}
              <XAxis 
                dataKey={config.xAxisColumn} 
                angle={-45} 
                textAnchor="end"
                interval={0}
                height={70}
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.xAxisColumn} 
                  position="bottom" 
                  offset={50}
                  style={labelStyle}
                />
              </XAxis>
              <YAxis 
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.yAxisColumns[0]} 
                  position="left" 
                  angle={-90} 
                  offset={-20}
                  style={labelStyle}
                />
              </YAxis>
              {config.showTooltip && <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: 'none' }} />}
              {config.showLegend && <Legend />}
              {config.yAxisColumns.map((column, index) => (
                <Line 
                  key={column} 
                  type="monotone" 
                  dataKey={column} 
                  stroke={colors[index % colors.length]} 
                  isAnimationActive={config.animation}
                  dot={{ stroke: colors[index % colors.length], strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )
        
      case 'scatter':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart
              margin={{ top: 20, right: 30, left: 40, bottom: 70 }}
            >
              {config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />}
              <XAxis 
                type="number" 
                dataKey={config.xAxisColumn} 
                name={config.xAxisColumn}
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.xAxisColumn} 
                  position="bottom" 
                  offset={50}
                  style={labelStyle}
                />
              </XAxis>
              <YAxis 
                type="number" 
                dataKey={config.yAxisColumns[0]} 
                name={config.yAxisColumns[0]}
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.yAxisColumns[0]} 
                  position="left" 
                  angle={-90} 
                  offset={-20}
                  style={labelStyle}
                />
              </YAxis>
              {config.showTooltip && <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: 'none' }} cursor={{ strokeDasharray: '3 3' }} />}
              {config.showLegend && <Legend />}
              <Scatter 
                name={config.yAxisColumns[0]} 
                data={processedData} 
                fill={colors[0]}
                isAnimationActive={config.animation}
              />
            </ScatterChart>
          </ResponsiveContainer>
        )
        
      case 'pie':
      case 'donut':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={processedData}
                cx="50%"
                cy="50%"
                labelLine={true}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={config.type === 'pie' ? 150 : 150}
                innerRadius={config.type === 'donut' ? 100 : 0}
                fill="#8884d8"
                dataKey="value"
                isAnimationActive={config.animation}
              >
                {processedData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              {config.showTooltip && <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: 'none' }} />}
              {config.showLegend && <Legend />}
            </PieChart>
          </ResponsiveContainer>
        )
        
      case 'area':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 40, bottom: 70 }}
            >
              {config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />}
              <XAxis 
                dataKey={config.xAxisColumn} 
                angle={-45} 
                textAnchor="end"
                interval={0}
                height={70}
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.xAxisColumn} 
                  position="bottom" 
                  offset={50}
                  style={labelStyle}
                />
              </XAxis>
              <YAxis 
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.yAxisColumns[0]} 
                  position="left" 
                  angle={-90} 
                  offset={-20}
                  style={labelStyle}
                />
              </YAxis>
              {config.showTooltip && <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: 'none' }} />}
              {config.showLegend && <Legend />}
              {config.yAxisColumns.map((column, index) => (
                <Area 
                  key={column} 
                  type="monotone" 
                  dataKey={column} 
                  stackId={config.stacked ? "1" : undefined}
                  fill={colors[index % colors.length]} 
                  stroke={colors[index % colors.length]}
                  isAnimationActive={config.animation}
                  fillOpacity={0.6}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        )
        
      case 'histogram':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 40, bottom: 70 }}
            >
              {config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />}
              <XAxis 
                dataKey={config.xAxisColumn} 
                angle={-45} 
                textAnchor="end"
                interval={0}
                height={70}
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value={config.xAxisColumn} 
                  position="bottom" 
                  offset={50}
                  style={labelStyle}
                />
              </XAxis>
              <YAxis 
                stroke="currentColor"
                style={axisStyle}
              >
                <Label 
                  value="Frequency" 
                  position="left" 
                  angle={-90} 
                  offset={-20}
                  style={labelStyle}
                />
              </YAxis>
              {config.showTooltip && <Tooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: 'none' }} />}
              {config.showLegend && <Legend />}
              <Bar 
                dataKey={config.xAxisColumn} 
                fill={colors[0]} 
                isAnimationActive={config.animation}
              />
            </BarChart>
          </ResponsiveContainer>
        )
        
      default:
        return <div>Chart type not supported</div>
    }
  }
  
  return (
    <div className="h-full w-full">
      {renderChart()}
    </div>
  )
}