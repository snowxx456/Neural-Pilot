import pandas as pd
import json
import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Configuration - Set your default file path here
DEFAULT_FILE_PATH = "./cleaned.csv"  # Change this to your default dataset path

class TargetColumnRecommender:
    def __init__(self,file_path):
        self.df = None
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.target_input = None
        self.problem_type = None
        self.recommendation_reason = None
        self.file_path = file_path

    def load_data(self):
        """Load and validate data from file."""
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(self.file_path)
            else:
                self.df = pd.read_csv(self.file_path)  # Try CSV as default

            # Clean column names
            self.df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '') 
                               for col in self.df.columns]

            print(f"\nData loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True

        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            return False

    def _validate_data(self):
        """Perform basic data validation checks."""
        if len(self.df) == 0:
            print("Warning: Dataset is empty!")
            return False

        missing_vals = self.df.isna().sum()
        if missing_vals.sum() > 0:
            print("\nMissing values detected:")
            print(missing_vals[missing_vals > 0])

        constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_cols:
            print("\nConstant columns (no variance):", constant_cols)

        return True

    def analyze_for_target(self):
        """Analyze dataset and recommend target column using Groq AI."""
        if self.df is None or len(self.df) == 0:
            print("No valid data loaded. Please load data first.")
            return None

        print("\nAnalyzing dataset to recommend best target column...")

        # Prepare dataset summary for AI analysis
        summary = {
            "columns": [],
            "sample_size": len(self.df),
            "missing_values": int(self.df.isna().sum().sum()),
            "potential_targets": []
        }

        for col in self.df.columns:
            col_info = {
                "name": col,
                "dtype": str(self.df[col].dtype),
                "unique_values": int(self.df[col].nunique()),
                "missing_percent": round(float(self.df[col].isna().mean() * 100), 1),
            }

            if self.df[col].dtype == 'object' and self.df[col].nunique() < 10:
                col_info["sample_values"] = self.df[col].unique().tolist()[:5]

            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_info.update({
                    "mean": round(float(self.df[col].mean()), 2),
                    "min": round(float(self.df[col].min()), 2),
                    "max": round(float(self.df[col].max()), 2),
                    "std_dev": round(float(self.df[col].std()), 2)
                })

            summary["columns"].append(col_info)

            if self._is_potential_target(col):
                summary["potential_targets"].append(col)

        # Generate AI prompt
        prompt = f"""Analyze this dataset summary and recommend the best target column for machine learning.
        Consider these factors in priority order:
        1. Business relevance (which column represents what we want to predict)
        2. Data quality (missing values, variance)
        3. Problem type suitability (classification vs regression)
        4. Column distribution and characteristics

        Dataset Summary:
        {json.dumps(summary, indent=2)}

        Respond with a JSON structure containing:
        {{
            "target_column": "column_name",
            "reason": "detailed justification",
            "problem_type": "regression/classification",
            "confidence_score": 0-100,
            "alternative_targets": ["col1", "col2"]
        }}"""

        try:
            print("\nConsulting Groq AI for target recommendation...")
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            if not all(k in result for k in ["target_column", "reason", "problem_type"]):
                raise ValueError("Invalid response format from LLM")

            print(f"\nRecommendation Results:")
            print(f"Target Column: {result['target_column']}")
            print(f"Problem Type: {result['problem_type']}")
            print(f"Confidence: {result.get('confidence_score', 'N/A')}/100")
            print(f"\nReasoning:\n{result['reason']}")

            # Save results
            self.target_input = result['target_column']
            self.problem_type = result['problem_type']
            self.recommendation_reason = result['reason']

        except Exception as e:
            print(f"\nError analyzing data with Groq LLM: {str(e)}")
            return None

    def _is_potential_target(self, col):
        """Determine if a column could be a good target."""
        if self.df[col].isna().mean() > 0.5:  # More than 50% missing
            return False
        if self.df[col].nunique() == 1:  # Constant value
            return False
        if self.df[col].nunique() == len(self.df) and self.df[col].dtype in ['int64', 'float64']:
            return False  # Probably an ID column
        return True
    def get_target_column(self):
        """Get the recommended target column."""
        return self.target_input if self.target_input else None
    def get_recommendation_reason(self):
        """Get the recommendation reason."""
        return self.recommendation_reason if self.recommendation_reason else None
    def get_file_path(self):
        """Get the file path."""
        return self.file_path if self.file_path else None

# def initialize(file_path=DEFAULT_FILE_PATH):
#     """Initialize and get target recommendation."""
#     recommender = TargetColumnRecommender()
#     if recommender.load_data(file_path):
#         result = recommender.analyze_for_target()
#         if result:
#             return {
#                 'file_path': file_path,
#                 'target_column': result['target_column'],
#                 'problem_type': result['problem_type'],
#                 'reason': result['reason']
#             }
#     return None

# # Initialize and export variables
# config = initialize()
# file_path = config['file_path'] if config else DEFAULT_FILE_PATH
# target_column = config['target_column'] if config else None