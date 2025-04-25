import { DatasetType } from '@/lib/types'
import { BarChart } from 'lucide-react'

interface DataHeaderProps {
  dataset: DatasetType | null
}

export function DataHeader({ dataset }: DataHeaderProps) {
  const rowCount = dataset?.rowCount || 0
  const columnCount = dataset?.columns?.length || 0
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
            {rowCount.toLocaleString()} rowCount · {typeof columnCount === 'number' ? columnCount : 0} columns
          </p>
        )}
      </div>
    </div>
  )
}