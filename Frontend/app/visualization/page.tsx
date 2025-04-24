'use client'

import { useState } from 'react'
import { FileUploader } from '@/components/visualization/FileUploader'
import { VisualizationDashboard } from '@/components/visualization/VisualizationDashboard'
import { DatasetType } from '@/lib/types'
import { DataHeader } from '@/components/visualization/DataHeader'

export default function Home() {
  const [dataset, setDataset] = useState<DatasetType | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <DataHeader dataset={dataset} />
        
        {!dataset ? (
          <div className="flex items-center justify-center h-[70vh]">
            <FileUploader 
              datasetId="default-dataset-id" // Replace with appropriate logic to generate or fetch datasetId
              onDatasetReady={setDataset}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
              error={error}
              setError={setError}
            />
          </div>            
        ) : (
          <VisualizationDashboard 
            dataset={dataset} 
            onReset={() => setDataset(null)}
          />
        )}
      </div>
    </main>
  )
}