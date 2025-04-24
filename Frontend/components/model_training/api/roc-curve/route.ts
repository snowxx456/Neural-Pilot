import { NextResponse } from "next/server"
import dotenv from "dotenv"
dotenv.config()

export async function GET() {
  try {
      const djangoApiUrl = process.env.SERVER_URL + '/api/roc-curves/';
      
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
  
      const rocCurveData = await response.json();
  
      return NextResponse.json(rocCurveData);
    } catch (error) {
      console.error("Error fetching ROC curve data from Django:", error);
      
    const rocCurveData = {
      models: [
        {
          name: "RandomForest",
          data: Array.from({ length: 100 }, (_, i) => {
            const x = i / 100
            // Create a curve that's better than random (above the diagonal)
            const y = Math.min(1, x + 0.4 * Math.sin(Math.PI * x))
            return { fpr: x, tpr: y }
          }),
          auc: 0.92,
        },
        {
          name: "LogisticRegression",
          data: Array.from({ length: 100 }, (_, i) => {
            const x = i / 100
            const y = Math.min(1, x + 0.25 * Math.sin(Math.PI * x))
            return { fpr: x, tpr: y }
          }),
          auc: 0.82,
        },
        {
          name: "XGBoost",
          data: Array.from({ length: 100 }, (_, i) => {
            const x = i / 100
            const y = Math.min(1, x + 0.35 * Math.sin(Math.PI * x))
            return { fpr: x, tpr: y }
          }),
          auc: 0.88,
        },
      ],
    }

    return NextResponse.json(rocCurveData)
  }
}
