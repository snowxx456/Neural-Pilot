'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Download, Share2, ChartBar, LineChart, PieChart, ChartScatter, AreaChart, BarChart3 } from 'lucide-react'
import { ChartConfig, ColumnMetadata, ChartType } from '@/lib/types'
import { ScrollArea } from '@/components/ui/scroll-area'

interface ChartControlsProps {
  config: ChartConfig;
  setConfig: (config: ChartConfig) => void;
  columns: Record<string, ColumnMetadata>;
}

export function ChartControls({ config, setConfig, columns }: ChartControlsProps) {
  const [shareLink, setShareLink] = useState<string | null>(null)
  
  // Available chart types with labels and icons
  const chartTypes: { value: ChartType; label: string; icon: any }[] = [
    { value: 'bar', label: 'Bar Chart', icon: ChartBar },
    { value: 'line', label: 'Line Chart', icon: LineChart },
    { value: 'scatter', label: 'Scatter Plot', icon: ChartScatter },
    { value: 'pie', label: 'Pie Chart', icon: PieChart },
    { value: 'area', label: 'Area Chart', icon: AreaChart },
    { value: 'histogram', label: 'Histogram', icon: BarChart3 }
  ]
  
  const handleChartTypeChange = (value: string) => {
    setConfig({
      ...config,
      type: value as ChartType
    })
  }
  
  const handleXAxisChange = (value: string) => {
    setConfig({
      ...config,
      xAxisColumn: value
    })
  }
  
  const handleYAxisChange = (value: string) => {
    setConfig({
      ...config,
      yAxisColumns: [value]
    })
  }
  
  const handleToggleOption = (option: keyof ChartConfig) => {
    setConfig({
      ...config,
      [option]: !config[option as keyof ChartConfig]
    })
  }
  
  const handleDownload = () => {
    alert('Chart download functionality would be implemented here.')
  }
  
  const handleShare = () => {
    const link = `https://data-viz-app.example/share/${Math.random().toString(36).substring(2, 10)}`
    setShareLink(link)
    navigator.clipboard.writeText(link)
      .then(() => alert('Link copied to clipboard.'))
      .catch(() => alert('Failed to copy link.'))
  }
  
  // Convert columns object to array and filter based on axis type
  const getColumnsForAxis = (axisType: 'x' | 'y'): ColumnMetadata[] => {
    return Object.values(columns).filter(column => {
      if (axisType === 'x') {
        // X-axis can be numeric or categorical
        return column.numeric || column.categorical;
      } else {
        // Y-axis should be numeric for most chart types
        return column.numeric;
      }
    });
  };

  return (
    <Card className="glass neon-border glow">
      <CardHeader>
        <CardTitle className="gradient-text text-xl">Chart Options</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[calc(100vh-300px)] pr-4">
          <div className="space-y-6">
            <div className="space-y-4">
              <Label className="text-sm font-medium opacity-70">Chart Type</Label>
              <div className="grid grid-cols-2 gap-2">
                {chartTypes.map(type => (
                  <Button
                    key={type.value}
                    variant={config.type === type.value ? "secondary" : "ghost"}
                    className={`flex flex-col items-center justify-center p-4 h-auto gap-2 transition-all ${
                      config.type === type.value ? 'neon-border' : ''
                    }`}
                    onClick={() => handleChartTypeChange(type.value)}
                  >
                    <type.icon className="h-6 w-6" />
                    <span className="text-xs">{type.label}</span>
                  </Button>
                ))}
              </div>
            </div>
            
            <Separator className="opacity-10" />
            
            {config.type !== 'pie' && config.type !== 'donut' && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-sm font-medium opacity-70">X-Axis</Label>
                  <Select 
                    value={config.xAxisColumn} 
                    onValueChange={handleXAxisChange}
                  >
                    <SelectTrigger className="glass">
                      <SelectValue placeholder="Select column" />
                    </SelectTrigger>
                    <SelectContent>
                      {getColumnsForAxis('x').map(column => (
                        <SelectItem key={column.name} value={column.name}>
                          {column.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                
                {(config.type !== 'histogram') && (
                  <div className="space-y-2">
                    <Label className="text-sm font-medium opacity-70">Y-Axis</Label>
                    <Select 
                      value={config.yAxisColumns[0] || ''} 
                      onValueChange={handleYAxisChange}
                    >
                      <SelectTrigger className="glass">
                        <SelectValue placeholder="Select column" />
                      </SelectTrigger>
                      <SelectContent>
                        {getColumnsForAxis('y').map(column => (
                          <SelectItem key={column.name} value={column.name}>
                            {column.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            )}
            
            <Separator className="opacity-10" />
            
            <div className="space-y-4">
              <Label className="text-sm font-medium opacity-70">Appearance</Label>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label htmlFor="show-legend" className="opacity-70">Legend</Label>
                  <Switch 
                    id="show-legend" 
                    checked={config.showLegend} 
                    onCheckedChange={() => handleToggleOption('showLegend')} 
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label htmlFor="show-grid" className="opacity-70">Grid Lines</Label>
                  <Switch 
                    id="show-grid" 
                    checked={config.showGrid} 
                    onCheckedChange={() => handleToggleOption('showGrid')} 
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label htmlFor="show-tooltip" className="opacity-70">Tooltips</Label>
                  <Switch 
                    id="show-tooltip" 
                    checked={config.showTooltip} 
                    onCheckedChange={() => handleToggleOption('showTooltip')} 
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label htmlFor="animation" className="opacity-70">Animations</Label>
                  <Switch 
                    id="animation" 
                    checked={config.animation} 
                    onCheckedChange={() => handleToggleOption('animation')} 
                  />
                </div>
                
                {(config.type === 'bar' || config.type === 'area') && (
                  <div className="flex items-center justify-between">
                    <Label htmlFor="stacked" className="opacity-70">Stacked</Label>
                    <Switch 
                      id="stacked" 
                      checked={config.stacked} 
                      onCheckedChange={() => handleToggleOption('stacked')} 
                    />
                  </div>
                )}
              </div>
            </div>
            
            <Separator className="opacity-10" />
            
            <div className="flex gap-2">
              <Button 
                variant="secondary" 
                className="w-full glass"
                onClick={handleDownload}
              >
                <Download className="mr-2 h-4 w-4" />
                Download
              </Button>
              <Button 
                variant="secondary" 
                className="w-full glass"
                onClick={handleShare}
              >
                <Share2 className="mr-2 h-4 w-4" />
                Share
              </Button>
            </div>
            
            {shareLink && (
              <div className="text-xs text-muted-foreground mt-2 break-all">
                Link copied: {shareLink}
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}