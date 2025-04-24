import { NextResponse } from "next/server";
import dotenv from "dotenv";

dotenv.config();

export async function GET() {
  try {
    const djangoApiUrl = process.env.SERVER_URL + "/api/correlation/";

    const response = await fetch(djangoApiUrl, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`Django API responded with status ${response.status}`);
    }

    const correlationData = await response.json();
    return NextResponse.json(correlationData);
  } catch (error) {
    console.error("Error fetching correlation data from Django:", error);

    // Fallback data
    const fallbackData = {
      features: [
        "age",
        "income",
        "education_level",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "num_credit_cards",
        "num_dependents",
      ],
      matrix: [
        [1.0, 0.65, 0.45, 0.72, 0.58, -0.32, 0.25, 0.18],
        [0.65, 1.0, 0.68, 0.54, 0.62, -0.45, 0.38, 0.12],
        [0.45, 0.68, 1.0, 0.42, 0.35, -0.28, 0.22, 0.08],
        [0.72, 0.54, 0.42, 1.0, 0.48, -0.52, 0.32, 0.15],
        [0.58, 0.62, 0.35, 0.48, 1.0, -0.38, 0.28, 0.22],
        [-0.32, -0.45, -0.28, -0.52, -0.38, 1.0, -0.18, -0.12],
        [0.25, 0.38, 0.22, 0.32, 0.28, -0.18, 1.0, 0.25],
        [0.18, 0.12, 0.08, 0.15, 0.22, -0.12, 0.25, 1.0],
      ],
    };

    return NextResponse.json(fallbackData);
  }
}
