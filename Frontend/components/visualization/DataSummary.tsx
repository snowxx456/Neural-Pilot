'use client'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { DatasetType } from '@/lib/types'
import { formatValue } from '@/lib/formatters'

interface DataSummaryProps {
  dataset: DatasetType
}

export function DataSummary({ dataset }: DataSummaryProps) {
  // Get type color based on column type
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'numeric':
        return 'bg-chart-1 text-primary-foreground'
      case 'categorical':
        return 'bg-chart-2 text-primary-foreground'
      case 'datetime':
        return 'bg-chart-3 text-primary-foreground'
      case 'boolean':
        return 'bg-chart-4 text-primary-foreground'
      default:
        return 'bg-muted text-muted-foreground'
    }
  }
  
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Dataset Summary</CardTitle>
          <CardDescription>
            Overview of dataset structure and content
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-card rounded-lg p-4 border">
              <div className="text-sm text-muted-foreground">Rows</div>
              <div className="text-2xl font-bold">{dataset.rows}</div>
            </div>
            <div className="bg-card rounded-lg p-4 border">
              <div className="text-sm text-muted-foreground">Columns</div>
              <div className="text-2xl font-bold">{dataset.columns.length}</div>
            </div>
            <div className="bg-card rounded-lg p-4 border">
              <div className="text-sm text-muted-foreground">Data Points</div>
              <div className="text-2xl font-bold">{dataset.rows * dataset.columns.length}</div>
            </div>
          </div>
          
          <div className="text-sm text-muted-foreground">
            {dataset.summary}
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Column Details</CardTitle>
          <CardDescription>
            Information about each column in the dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Unique Values</TableHead>
                  <TableHead>Missing</TableHead>
                  <TableHead>Min</TableHead>
                  <TableHead>Max</TableHead>
                  <TableHead>Mean/Mode</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dataset.columns.map((column) => (
                  <TableRow key={column.name}>
                    <TableCell className="font-medium">{column.name}</TableCell>
                    <TableCell>
                      <Badge className={getTypeColor(column.type)}>
                        {column.type}
                      </Badge>
                    </TableCell>
                    <TableCell>{column.uniqueValues ?? 'N/A'}</TableCell>
                    <TableCell>{column.missing ?? 0}</TableCell>
                    <TableCell>{formatValue(column.min)}</TableCell>
                    <TableCell>{formatValue(column.max)}</TableCell>
                    <TableCell>
                      {column.type === 'numeric' 
                        ? formatValue(column.mean)
                        : formatValue(column.mode)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Data Preview</CardTitle>
          <CardDescription>
            First 10 rows of the dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <div className="w-full overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    {dataset.columns.map((column) => (
                      <TableHead key={column.name}>{column.name}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {dataset.data.slice(0, 10).map((row, i) => (
                    <TableRow key={i}>
                      {dataset.columns.map((column) => (
                        <TableCell key={`${i}-${column.name}`}>
                          {formatValue(row[column.name])}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}