// Format values for display
export const formatValue = (value: any): string => {
  if (value === null || value === undefined || value === '') {
    return 'N/A';
  }

  // Handle special numeric values
  if (typeof value === 'string') {
    if (value === 'Infinity') return '∞';
    if (value === '-Infinity') return '-∞';
    if (value === 'NaN') return 'N/A';
  }
  
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return 'N/A';
    }
    // Format numbers with 2 decimal places if they have decimals
    return Number.isInteger(value) ? 
      value.toString() : 
      value.toFixed(2);
  }
  
  if (typeof value === 'boolean') {
    return value.toString();
  }
  
  if (value instanceof Date) {
    return value.toLocaleDateString();
  }
  
  return String(value);
}

// Format dates for display
export function formatDate(date: string | Date): string {
  if (!date) return 'N/A'
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date
    return dateObj.toLocaleDateString()
  } catch (e) {
    return 'Invalid Date'
  }
}

// Format numbers with commas for thousands
export function formatNumber(num: number): string {
  if (num === undefined || num === null) return 'N/A'
  
  return new Intl.NumberFormat().format(num)
}