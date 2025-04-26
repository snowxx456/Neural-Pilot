"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DatasetType } from "@/lib/types";
import { formatValue } from "@/lib/formatters";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";

interface DataSummaryProps {
  dataset: DatasetType;
}

export function DataSummary({ dataset }: DataSummaryProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [visualData, setVisualData] = useState<DatasetType | null>(null);
  const API = process.env.NEXT_PUBLIC_SERVER_URL || "http://localhost:8000/";

  useEffect(() => {
    const fetchVisualizationData = async () => {
      const storedData = localStorage.getItem("selectedDataset");
      if (!storedData) {
        setError("No dataset information found");
        setLoading(false);
        return;
      }

      try {
        const { id } = JSON.parse(storedData);
        setLoading(true);
        setError(null);

        const response = await fetch(`${API}api/visualization/${id}/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error("Dataset not found on server");
          }
          throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();
        if (!data || typeof data !== "object") {
          throw new Error("Invalid data format received from API");
        }
        console.log("Visualization data:", data);

        setVisualData(data);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to load data";
        setError(errorMessage);
        console.error("Error loading visualization:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchVisualizationData();
  }, []);

  const getTypeColor = (type: string) => {
    switch (type) {
      case "numeric":
        return "bg-blue-500 text-white";
      case "categorical":
        return "bg-green-500 text-white";
      case "datetime":
        return "bg-purple-500 text-white";
      case "boolean":
        return "bg-yellow-500 text-white";
      default:
        return "bg-gray-500 text-white";
    }
  };

  const isValidNumber = (value: any): boolean => {
    return typeof value === "number" && Number.isFinite(value);
  };

  const calculateDataPoints = (): number => {
    const rows = visualData?.rowCount || 0;
    const cols = visualData?.columns
      ? Object.keys(visualData.columns).length
      : 0;
    return rows * cols;
  };

  if (loading) {
    return <DataSummarySkeleton />;
  }

  if (error) {
    return (
      <div className="space-y-4">
        <Alert variant="destructive">
          <AlertTitle>Error Loading Data</AlertTitle>
          <AlertDescription className="flex flex-col gap-3">
            <div>{error}</div>
            <Button
              variant="outline"
              onClick={() => window.location.reload()}
              className="w-fit"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
        <DataSummarySkeleton />
      </div>
    );
  }

  if (
    !visualData ||
    !visualData.columns ||
    Object.keys(visualData.columns).length === 0
  ) {
    return (
      <Alert>
        <AlertDescription>
          Dataset loaded but contains no column information. Try refreshing.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Dataset Summary Card */}
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
              <div className="text-2xl font-bold">
                {isValidNumber(visualData.rowCount)
                  ? visualData.rowCount.toLocaleString()
                  : "N/A"}
              </div>
            </div>
            <div className="bg-card rounded-lg p-4 border">
              <div className="text-sm text-muted-foreground">Columns</div>
              <div className="text-2xl font-bold">
                {Object.keys(visualData.columns).length}
              </div>
            </div>
            <div className="bg-card rounded-lg p-4 border">
              <div className="text-sm text-muted-foreground">Data Points</div>
              <div className="text-2xl font-bold">
                {calculateDataPoints().toLocaleString()}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Column Details Card */}
      <Card>
        <CardHeader>
          <CardTitle>Column Details</CardTitle>
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
                {Object.entries(visualData.columns).map(([key, column]) => (
                  <TableRow key={key}>
                    <TableCell className="font-medium">{column.name}</TableCell>
                    <TableCell>
                      <Badge className={getTypeColor(column.type)}>
                        {column.type}
                      </Badge>
                    </TableCell>
                    <TableCell>{column.unique ?? "N/A"}</TableCell>
                    <TableCell>{column.missing ?? 0}</TableCell>
                    <TableCell>{formatValue(column.min)}</TableCell>
                    <TableCell>{formatValue(column.max)}</TableCell>
                    <TableCell>
                      {column.type === "numeric"
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

      {/* Data Preview Card */}
      <Card>
        <CardHeader>
          <CardTitle>Data Preview</CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <Table>
              <TableHeader>
                <TableRow>
                  {Object.values(visualData.columns).map((column) => (
                    <TableHead key={column.name}>{column.name}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {(visualData.data || []).slice(0, 10).map((row, i) => (
                  <TableRow key={i}>
                    {Object.keys(visualData.columns).map((columnKey) => (
                      <TableCell key={`${i}-${columnKey}`}>
                        {formatValue(row[columnKey])}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

function DataSummarySkeleton() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-[200px]" />
          <Skeleton className="h-4 w-[300px]" />
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-card rounded-lg p-4 border">
              <Skeleton className="h-4 w-[100px] mb-2" />
              <Skeleton className="h-8 w-[120px]" />
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-[200px]" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[400px]" />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-[200px]" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[400px]" />
        </CardContent>
      </Card>
    </div>
  );
}
