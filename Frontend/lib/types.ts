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
export type ColumnType = 'string' | 'number' | 'boolean' | 'date';

// Chart types that can be recommended
export type ChartType = 'bar' | 'line' | 'pie' | 'scatter' | 'donut';

// A column in a dataset with metadata
export interface ColumnMetadata {
  name: string;
  type: ColumnType;
  unique: number;
  missing: number;
  numeric: boolean;
  categorical: boolean;
}

// A chart recommendation based on data characteristics
export interface ChartRecommendation {
  type: ChartType;
  title: string;
  description: string;
  xAxis?: string;
  yAxis?: string[];
  confidence: number;
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
}