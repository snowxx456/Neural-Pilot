// Format values for display
export function formatValue(value: any): string {
  if (value === undefined || value === null) {
    return 'N/A'
  }
  
  if (typeof value === 'number') {
    // Round to 2 decimal places if it's a float
    return Number.isInteger(value) ? value.toString() : value.toFixed(2)
  }
  
  if (value instanceof Date) {
    return value.toLocaleDateString()
  }
  
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False'
  }
  
  return String(value)
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