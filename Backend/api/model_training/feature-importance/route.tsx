import { NextResponse } from "next/server"
import type { FeatureImportance } from "@/lib/types"

// Mock feature importance data
const featureImportance: FeatureImportance[] = [
  { feature: "age", importance: 0.185 },
  { feature: "income", importance: 0.162 },
  { feature: "education_level", importance: 0.143 },
  { feature: "credit_score", importance: 0.128 },
  { feature: "employment_years", importance: 0.112 },
  { feature: "debt_to_income", importance: 0.098 },
  { feature: "num_credit_cards", importance: 0.087 },
  { feature: "num_dependents", importance: 0.076 },
  { feature: "has_mortgage", importance: 0.065 },
  { feature: "has_car_loan", importance: 0.054 },
  { feature: "savings_amount", importance: 0.043 },
  { feature: "is_homeowner", importance: 0.032 },
  { feature: "years_at_residence", importance: 0.025 },
  { feature: "has_student_loan", importance: 0.018 },
  { feature: "marital_status", importance: 0.012 },
  { feature: "gender", importance: 0.008 },
  { feature: "zipcode", importance: 0.005 },
  { feature: "state", importance: 0.003 },
]

export async function GET() {
  return NextResponse.json(featureImportance)
}
