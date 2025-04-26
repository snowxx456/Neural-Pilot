import type {
  ModelResult,
  FeatureImportance,
  ConfusionMatrixData,
} from "./types";
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/";
let id: number | null = null;
let name: string | null = null;
const storedData = localStorage.getItem("selectedDataset");
if (storedData) {
  try {
    const data = JSON.parse(storedData);
    id = data.id;
    name = data.name;
    console.log("Dataset ID:", data.id);
  } catch (error) {
    console.error("Error parsing dataset from localStorage:", error);
  }
}

// Fetch model results from the API
export async function fetchModelResults(): Promise<ModelResult[]> {
  console.log(id);

  if (!id) {
    console.error("Model ID not found in local storage");
    // Return mock data as fallback when model ID is not found
    return [];
  }

  try {
    // Use a more specific API endpoint for getting model results
    const response = await fetch(`${API}api/models/${id}/`);

    console.log("Model results fetched successfully:", response);

    if (!response.ok) {
      console.error(`API error: ${response.status} ${response.statusText}`);
      // Return mock data as fallback when API fails
      return [];
    }

    return response.json();
  } catch (error) {
    console.error("Error fetching model results:", error);
    // Return mock data as fallback when fetch fails
    return [];
  }
}

export async function startModelTraining(
  id: string
): Promise<{ success: boolean }> {
  const response = await fetch(`/api/train/${id}/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error("Failed to start model training");
  }

  return { success: true };
}

// Fetch feature importance data
export async function fetchFeatureImportance(): Promise<FeatureImportance[]> {
  const response = await fetch(`${API}api/feature-importance/${id}/`);
  if (!response.ok) {
    throw new Error("Failed to fetch feature importance data");
  }
  return response.json();
}

// Fetch confusion matrix data
export async function fetchConfusionMatrix(): Promise<ConfusionMatrixData> {
  const response = await fetch(`${API}api/confusionmatrix/${id}/`);
  if (!response.ok) {
    throw new Error("Failed to fetch confusion matrix data");
  }
  return response.json();
}

// Fetch correlation matrix data
export async function fetchCorrelationMatrix() {
  const response = await fetch(`${API}api/correlation/${id}/`);
  if (!response.ok) {
    throw new Error("Failed to fetch correlation matrix data");
  }
  return response.json();
}

// Fetch ROC curve data
export async function fetchRocCurveData() {
  const response = await fetch(`${API}api/roc-curve/${id}/`);
  if (!response.ok) {
    throw new Error("Failed to fetch ROC curve data");
  }
  return response.json();
}

// Fetch precision-recall curve data
export async function fetchPrecisionRecallCurveData() {
  const response = await fetch(`${API}api/precision-recall/${id}/`);
  if (!response.ok) {
    throw new Error("Failed to fetch precision-recall curve data");
  }
  return response.json();
}

// Estimate training time based on dataset size and model complexity
export function estimateTrainingTime(
  datasetSize: number,
  modelComplexity: "low" | "medium" | "high"
): number {
  // Base time in seconds
  let baseTime = 0;

  // Adjust based on model complexity
  switch (modelComplexity) {
    case "low":
      baseTime = 5;
      break;
    case "medium":
      baseTime = 15;
      break;
    case "high":
      baseTime = 30;
      break;
  }

  // Scale based on dataset size (simplified estimation)
  const scaleFactor = datasetSize / 1000;

  // Return estimated time in seconds
  return Math.round(baseTime * scaleFactor);
}
