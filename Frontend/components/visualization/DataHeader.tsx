import { DatasetType } from '@/lib/types'
import { BarChart } from 'lucide-react'

interface DataHeaderProps {
  dataset: DatasetType | null
}

export function DataHeader({ dataset }: DataHeaderProps) {
  // Getting row and column count from stats object, or calculate from data if not in stats
  const rowCount = dataset?.rowCount || 0
  const columnCount = dataset?.columns ? Object.keys(dataset?.columns).length : 0
  
  const datasetName = dataset?.filename || 'Data Visualization System'
  
  return (
    <div className="flex items-center gap-3 mb-8">
      <div className="bg-primary/10 p-2 rounded-md">
        <BarChart className="h-6 w-6 text-primary" />
      </div>
      <div>
        <h1 className="text-2xl font-bold">
          {datasetName}
        </h1>
        {dataset && (
          <p className="text-sm text-muted-foreground">
            {rowCount.toLocaleString()} rows · {columnCount.toLocaleString()} columns 
          </p>
        )}
      </div>
    </div>
  )
}