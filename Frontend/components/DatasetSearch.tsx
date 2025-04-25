import { useState } from 'react';
import { useToast } from '@/hooks/use-toast';
import axios from 'axios';

interface Dataset {
  title: string;
  owner: string;
  description: string;
  size: string;
  downloads: number;
  lastUpdated: string;
  url: string;
  ref: string;
}

export default function DatasetSearch() {
  const [searchQuery, setSearchQuery] = useState('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/search/', {
        query: searchQuery
      });

      setDatasets(response.data.datasets);

      if (response.data.datasets.length === 0) {
        toast({
          title: "No datasets found",
          description: "Try a different search query",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to search datasets. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4 space-y-4">
      <form onSubmit={handleSearch} className="space-y-2">
        <div className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search for datasets..."
            className="flex-1 p-2 border rounded-md"
            disabled={loading}
            required
          />
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-blue-300"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      <div className="grid gap-4 md:grid-cols-2">
        {datasets.map((dataset, index) => (
          <div
            key={index}
            className="border rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow"
          >
            <h3 className="text-lg font-semibold mb-2">{dataset.title}</h3>
            <p className="text-sm text-gray-600 mb-3 line-clamp-2">
              {dataset.description}
            </p>
            <div className="grid grid-cols-2 gap-2 text-sm text-gray-500">
              <div>👤 {dataset.owner}</div>
              <div>⬇️ {dataset.downloads.toLocaleString()} downloads</div>
              <div>📅 {dataset.lastUpdated}</div>
              <div>💾 {dataset.size}</div>
            </div>
            <a
              href={dataset.url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 inline-flex items-center text-sm text-blue-500 hover:text-blue-600"
            >
              View on Kaggle
              <svg
                className="w-4 h-4 ml-1"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          </div>
        ))}
      </div>

      {datasets.length === 0 && !loading && searchQuery && (
        <div className="text-center text-gray-500 py-8">
          No datasets found. Try a different search term.
        </div>
      )}
    </div>
  );
}