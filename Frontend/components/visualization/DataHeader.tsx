import { DatasetType } from '@/lib/types'
import { BarChart } from 'lucide-react'

interface DataHeaderProps {
  dataset: DatasetType | null
}

export function DataHeader({ dataset }: DataHeaderProps) {
  return (
    <div className="flex items-center gap-3 mb-8">
      <div className="bg-primary/10 p-2 rounded-md">
        <BarChart className="h-6 w-6 text-primary" />
      </div>
      <div>
        <h1 className="text-2xl font-bold">
          {dataset ? dataset.name : 'Data Visualization System'}
        </h1>
        {dataset && (
          <p className="text-sm text-muted-foreground">
            {dataset.rows} rows · {dataset.columns.length} columns
          </p>
        )}
      </div>
    </div>
  )
}