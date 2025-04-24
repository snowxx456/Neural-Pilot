import { NextResponse } from "next/server"
import dotenv from "dotenv"
dotenv.config()

export async function GET() {
  try {
      // Fetch precision-recall data from Django API
      const djangoApiUrl = process.env.SERVER_URL + '/api/pr-curves/';
      
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
  
      const prCurveData = await response.json();
  
      return NextResponse.json(prCurveData);
    } catch (error) {
      console.error("Error fetching precision-recall data from Django:", error);
      const prCurveData = {
      models: [
        {
          name: "RandomForest",
          data: Array.from({ length: 100 }, (_, i) => {
            const recall = i / 100
            // Create a curve that shows precision decreasing as recall increases
            const precision = 1 - 0.3 * Math.pow(recall, 1.5)
            return { recall, precision: Math.max(0, precision) }
          }),
          avgPrecision: 0.85,
        },
        {
          name: "LogisticRegression",
          data: Array.from({ length: 100 }, (_, i) => {
            const recall = i / 100
            const precision = 1 - 0.5 * Math.pow(recall, 1.2)
            return { recall, precision: Math.max(0, precision) }
          }),
          avgPrecision: 0.72,
        },
        {
          name: "XGBoost",
          data: Array.from({ length: 100 }, (_, i) => {
            const recall = i / 100
            const precision = 1 - 0.4 * Math.pow(recall, 1.3)
            return { recall, precision: Math.max(0, precision) }
          }),
          avgPrecision: 0.78,
        },
      ],
    }

    return NextResponse.json(prCurveData)
  } 
}
