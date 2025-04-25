import { NextResponse } from "next/server"
import type { ConfusionMatrixData } from "@/lib/types"
import * as dotenv from "dotenv"

dotenv.config()

export async function GET() {
  try {
    // Fetch confusion matrix data from Django API
    const djangoApiUrl = process.env.SERVER_URL + '/api/confusionmatrix/';
    
    const response = await fetch(djangoApiUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store'
    });

    if (!response.ok) {
      throw new Error(`Django API responded with status ${response.status}`);
    }

    const confusionMatrix: ConfusionMatrixData = await response.json();

    return NextResponse.json(confusionMatrix);
  } catch (error) {
    console.error("Error fetching confusion matrix from Django:", error);
    
    // Fallback mock data
    const fallbackData: ConfusionMatrixData = {
      matrix: [
        [1180, 413],
        [142, 265],
      ],
      labels: ["0", "1"],
    };

    return NextResponse.json(fallbackData, {
      headers: {
        'X-API-Warning': 'Using fallback data due to Django API failure'
      }
    });
  }
}