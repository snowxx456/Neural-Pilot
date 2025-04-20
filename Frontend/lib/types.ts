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