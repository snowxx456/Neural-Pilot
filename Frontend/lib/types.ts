export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export interface Dataset {
  id: string;
  title: string;
  description: string;
  downloadLink: string;
  externalLink: string;
  category: string;
  ref: string;
  owner?: string;
  downloads?: number;
  lastUpdated?: string;
  size?: string;
}

export type PreprocessingStatus = 'idle' | 'processing' | 'completed' | 'error';

export interface PreprocessingStep {
  id: number;
  title: string;
  description: string;
  icon: any;
  status: 'pending' | 'processing' | 'completed';
}


// Column types that can be detected in data
export type ColumnType = 'string' | 'number' | 'boolean' | 'date' | 'datetime' |'numeric' | 'categorical'| 'boolean' ;

// Chart types that can be recommended
export type ChartType = 'bar' | 'line' | 'pie' | 'scatter' | 'donut' | 'area' | 'histogram';

// A column in a dataset with metadata
export interface ColumnMetadata {
  name: string;
  type: ColumnType;
  unique: number; // Changed from uniqueValues?
  missing: number;
  numeric: boolean; // Add this
  categorical: boolean; // Add this
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  mode?: string | number | null;
}

// A chart recommendation based on data characteristics
export interface ChartRecommendation {
  type: ChartType;
  title: string;
  description: string;
  xAxis?: string;
  yAxis?: string[];
  confidence: number;
  filter?: Record<string, any>;
}

// The processed dataset returned from the backend
export interface DatasetType {
  data: any[];
  columns: Record<string, ColumnMetadata>;
  recommendations: ChartRecommendation[];
  filename: string;
  rowCount: number;
  columnCount: number;
}

// Chart configuration for rendering
export interface ChartConfig {
  type: ChartType;
  title: string;
  xAxisColumn: string;
  yAxisColumns: string[];
  colors: string[];
  showLegend: boolean;
  showGrid: boolean;
  showTooltip: boolean;
  animation: boolean;
  stacked?: boolean;
  area?: boolean;
  filter?: Record<string, any>;
}

export interface ModelMetrics {
  precision: number
  recall: number
  f1: number
}

export interface CrossValidationScore {
  mean: number
  std: number
}

export interface ModelResult {
  name: string
  accuracy: number
  metrics: ModelMetrics
  report: string
  trainingTime: number
  cvTime: number
  cvScore: CrossValidationScore
  parameters: Record<string, any>
  description?: string
  isBest: boolean
}

export interface FeatureImportance {
  feature: string
  importance: number
}

export interface ConfusionMatrixData {
  matrix: number[][]
  labels: string[]
}

export interface FileUploadResponse {
  success: boolean;
  fileId?: string;
  error?: string;
  metadata?: {
    rows: number;
    columns: number;
    size: string;
  };
}

export interface ProcessingStep {
  id: string;
  name: string;
  description: string;
  completed: boolean;
  current: boolean;
}