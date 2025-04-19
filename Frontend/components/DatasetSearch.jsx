import { useState } from 'react';
import axios from 'axios';
import DatasetSearch from './components/DatasetSearch';

export default function DatasetSearch() {
    const [searchQuery, setSearchQuery] = useState('');
    const [datasets, setDatasets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSearch = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const response = await axios.post('http://localhost:8000/api/search/', {
                query: searchQuery
            });

            setDatasets(response.data.datasets);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to search datasets');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <form onSubmit={handleSearch}>
                <div className="flex gap-2 mb-4">
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search for datasets..."
                        className="flex-1 p-2 border rounded"
                        required
                    />
                    <button 
                        type="submit"
                        disabled={loading}
                        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
                    >
                        {loading ? 'Searching...' : 'Search'}
                    </button>
                </div>
            </form>

            {error && (
                <div className="text-red-500 mb-4">
                    {error}
                </div>
            )}

            <div className="grid gap-4">
                {datasets.map((dataset, index) => (
                    <div key={index} className="border rounded-lg p-4 shadow">
                        <h3 className="text-xl font-semibold mb-2">{dataset.title}</h3>
                        <p className="text-gray-600 mb-2">{dataset.description}</p>
                        <div className="grid grid-cols-2 gap-2 text-sm text-gray-500">
                            <div>👤 Owner: {dataset.owner}</div>
                            <div>⬇️ Downloads: {dataset.downloads}</div>
                            <div>📅 Last Updated: {dataset.lastUpdated}</div>
                            <div>💾 Size: {dataset.size}</div>
                        </div>
                        <a 
                            href={dataset.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mt-2 inline-block text-blue-500 hover:underline"
                        >
                            View on Kaggle →
                        </a>
                    </div>
                ))}
            </div>
        </div>
    );
}