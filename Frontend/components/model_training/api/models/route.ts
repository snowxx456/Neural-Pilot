import { NextResponse } from "next/server"
import type { ModelResult } from "@/lib/types"
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
    const modelResult: ModelResult = await response.json();
    return NextResponse.json(modelResult);
    
  } catch (error) {
    console.error("Error fetching model results:", error);
      const fallbackData: ModelResult[] = [
          {
            name: "LogisticRegression",
            accuracy: 0.724,
            metrics: {
              precision: 0.79,
              recall: 0.72,
              f1: 0.75,
            },
            report: `              precision    recall  f1-score   support

                0       0.89      0.74      0.81      1593
                1       0.39      0.65      0.49       407

          accuracy                           0.72      2000
        macro avg       0.64      0.70      0.65      2000
        weighted avg       0.79      0.72      0.75      2000`,
            trainingTime: 1.44,
            cvTime: 6.44,
            cvScore: {
              mean: 0.7146,
              std: 0.0063,
            },
            parameters: {
              C: 1.0,
              penalty: "l2",
              solver: "lbfgs",
              max_iter: 10000,
            },
            description: "Linear classifier with logistic function",
            isBest: false,
          },
          {
            name: "RandomForest",
            accuracy: 0.835,
            metrics: {
              precision: 0.84,
              recall: 0.83,
              f1: 0.83,
            },
            report: `              precision    recall  f1-score   support

                0       0.92      0.86      0.89      1593
                1       0.58      0.73      0.65       407

          accuracy                           0.84      2000
        macro avg       0.75      0.80      0.77      2000
        weighted avg       0.84      0.83      0.83      2000`,
            trainingTime: 3.21,
            cvTime: 12.65,
            cvScore: {
              mean: 0.8245,
              std: 0.0089,
            },
            parameters: {
              n_estimators: 100,
              max_depth: 15,
              min_samples_split: 5,
              min_samples_leaf: 2,
            },
            description: "Ensemble of decision trees",
            isBest: true,
          },
          {
            name: "GradientBoosting",
            accuracy: 0.815,
            metrics: {
              precision: 0.82,
              recall: 0.81,
              f1: 0.81,
            },
            report: `              precision    recall  f1-score   support

                0       0.91      0.84      0.87      1593
                1       0.55      0.71      0.62       407

          accuracy                           0.82      2000
        macro avg       0.73      0.78      0.75      2000
        weighted avg       0.82      0.81      0.81      2000`,
            trainingTime: 5.67,
            cvTime: 18.92,
            cvScore: {
              mean: 0.8032,
              std: 0.0102,
            },
            parameters: {
              n_estimators: 150,
              learning_rate: 0.1,
              max_depth: 5,
              subsample: 0.8,
            },
            description: "Boosting with gradient descent",
            isBest: false,
          },
          {
            name: "XGBoost",
            accuracy: 0.825,
            metrics: {
              precision: 0.83,
              recall: 0.82,
              f1: 0.82,
            },
            report: `              precision    recall  f1-score   support

                0       0.91      0.85      0.88      1593
                1       0.57      0.72      0.64       407

          accuracy                           0.83      2000
        macro avg       0.74      0.79      0.76      2000
        weighted avg       0.83      0.82      0.82      2000`,
            trainingTime: 2.89,
            cvTime: 10.45,
            cvScore: {
              mean: 0.8156,
              std: 0.0095,
            },
            parameters: {
              n_estimators: 200,
              learning_rate: 0.05,
              max_depth: 6,
              subsample: 0.9,
              colsample_bytree: 0.8,
            },
            description: "Extreme Gradient Boosting",
            isBest: false,
          },
          {
            name: "SVM",
            accuracy: 0.745,
            metrics: {
              precision: 0.76,
              recall: 0.74,
              f1: 0.75,
            },
            report: `              precision    recall  f1-score   support

                0       0.88      0.78      0.83      1593
                1       0.45      0.63      0.52       407

          accuracy                           0.74      2000
        macro avg       0.67      0.71      0.68      2000
        weighted avg       0.76      0.74      0.75      2000`,
            trainingTime: 8.32,
            cvTime: 25.67,
            cvScore: {
              mean: 0.7356,
              std: 0.0112,
            },
            parameters: {
              C: 1.0,
              kernel: "rbf",
              gamma: "scale",
            },
            description: "Support Vector Machine",
            isBest: false,
          },
          {
            name: "NaiveBayes",
            accuracy: 0.695,
            metrics: {
              precision: 0.71,
              recall: 0.69,
              f1: 0.7,
            },
            report: `              precision    recall  f1-score   support

                0       0.85      0.75      0.80      1593
                1       0.38      0.54      0.45       407

          accuracy                           0.70      2000
        macro avg       0.62      0.65      0.62      2000
        weighted avg       0.71      0.69      0.70      2000`,
            trainingTime: 0.12,
            cvTime: 0.89,
            cvScore: {
              mean: 0.6845,
              std: 0.0156,
            },
            parameters: {
              var_smoothing: 1e-9,
            },
            description: "Probabilistic classifier",
            isBest: false,
          },
        ]
  return NextResponse.json(fallbackData,{
    headers: {
      'X-API-Warning': 'Using fallback data due to Django API failure'
    }
  });
  }
}