# llm_cleaning.py
from langchain_groq import ChatGroq
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import os
from scipy import stats
from dotenv import load_dotenv
import logging
import json
import re
import warnings
from sklearn.preprocessing import (
    FunctionTransformer, 
    KBinsDiscretizer,
    OrdinalEncoder
)
from sklearn.experimental import enable_iterative_imputer  # Required for IterativeImputer
from sklearn.impute import IterativeImputer
from scipy.stats import zscore, iqr
from dateutil.parser import parse as date_parser
import re
import ast
import unicodedata
warnings.filterwarnings('ignore')
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_python_code(text):
    """Extract Python code from text that might contain markdown code blocks"""
    # Try to extract code from markdown code blocks first
    code_blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # If no markdown code blocks, look for code patterns
    code_blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # If still nothing, assume the whole text is code (if it looks like Python)
    if 'def ' in text and 'import ' in text:
        return text
    
    # Default fallback
    return text

class LLMCLEANINGAGENT:
    def __init__(self, input_file=None, df=None, llm_model="deepseek-r1-distill-llama-70b", target_col=None, cleaning_strategy="auto"):
        """
        Initialize the pipeline with input file or dataframe and parameters
        
        Args:
            input_file: Path to the CSV file to clean (optional if df is provided)
            df: Pandas DataFrame to clean (optional if input_file is provided)
            llm_model: The Groq LLM model to use
            target_col: Target column for supervised learning tasks
            cleaning_strategy: "auto", "aggressive", or "conservative"
        """
        self.input_file = input_file
        self.df = df
        self.llm_model = llm_model
        self.target_col = target_col
        self.cleaning_strategy = cleaning_strategy
        self.cleaned_df = None
        self.llm = None
        self.cleaning_code = None
        self.llm_suggestions = None
        self.summary = {}
        self.issues = {}
        self.shared_namespace = {}
    
    def load_data(self):
        """Load the data from the input file or use provided dataframe"""
        if self.df is not None:
            logger.info("Using provided DataFrame")
            return True
            
        if not self.input_file:
            logger.error("No input file provided and no DataFrame provided")
            return False
            
        logger.info(f"Reading the CSV file: {self.input_file}")
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Successfully loaded data with shape: {self.df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
        
    def generate_summary(self):
        """Generate comprehensive data profile"""
        summary = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing = self.df[col].isnull().sum()
            total = len(self.df)
            missing_pct = round((missing / total) * 100, 2)
            unique_count = self.df[col].nunique()
            
            # Convert to strings and limit sample size to avoid memory issues
            sample_vals = self.df[col].dropna().astype(str).head(10).tolist()

            stats = {
                "dtype": dtype,
                "missing_percent": float(missing_pct),
                "unique_count": int(unique_count),
                "sample_values": sample_vals
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats.update({
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "mean": float(self.df[col].mean()),
                    "skew": float(self.df[col].skew())
                })
            
            summary[col] = stats
        self.summary = summary
        logger.info("Data summary generated successfully")
        return summary
    
    def initialize_llm(self):
        """Initialize the Groq LLM"""
        logger.info(f"Initializing Groq LLM with model: {self.llm_model}")
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            if not groq_api_key:
                logger.error("GROQ_API_KEY not found in environment variables")
                raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
                
            self.llm = ChatGroq(
                model=self.llm_model,
                groq_api_key=groq_api_key
            )
            logger.info("Successfully initialized Groq LLM")
            return True
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {e}")
            return False
        
    def detect_issues(self):
        """Identify data quality issues automatically"""
        issues = {}
        for col, stats in self.summary.items():
            col_issues = []
            
            # Missing values
            if stats["missing_percent"] > 30:
                col_issues.append(("critical", "High missing values (>30%)"))
            elif stats["missing_percent"] > 5:
                col_issues.append(("warning", "Moderate missing values (>5%)"))
            
            # Numeric outliers
            if stats.get("skew", None) is not None:
                if abs(stats["skew"]) > 1:
                    col_issues.append(("warning", f"Skewed distribution (skew={stats['skew']:.2f})"))
            
            # High cardinality
            if stats["dtype"] == "object" and stats["unique_count"] > 50:
                col_issues.append(("warning", f"High cardinality ({stats['unique_count']} unique values)"))
            
            # Constant values
            if stats["unique_count"] == 1:
                col_issues.append(("critical", "Constant value column"))
            
            issues[col] = col_issues
        self.issues = issues
        logger.info("Data issues detected successfully")
        return issues
        
    def get_llm_cleaning_suggestions(self):
        """Get cleaning suggestions from LLM"""
        logger.info("Requesting cleaning suggestions from LLM")
        try:
            combined_data = {
                "summary": self.summary,
                "issues": self.issues
            }
            
            prompt = f"""
                Analyze this dataset sample and provide structured cleaning recommendations:

                SAMPLE DATA:
                {json.dumps(combined_data, indent=2)}

                Format your response as JSON with this structure:
                {{
                    "columns": {{
                        "column_name": {{
                            "type": "detected type (numeric/categorical/datetime)",
                            "anomalies": ["list of specific issues"],
                            "operations": ["specific cleaning steps needed"],
                            "example_fix": "example value transformation"
                        }}
                    }},
                    "global_issues": ["cross-column problems"],
                    "priority_order": ["recommended cleaning sequence"]
                }}

                Include these potential anomaly checks:
                - Check for data type
                - Invalid data types
                - Outliers (specify detection method)
                - Missing values (NaN, null, 0, -999)
                - Inconsistent formatting
                - Invalid categorical values
                - DateTime format mismatches
                - Duplicate records
                - Encoding issues
                - Numeric value range violations
                - Textual pattern mismatches
                """
            
            response = self.llm.invoke(prompt)
            self.llm_suggestions = response.content
            logger.info("Successfully received LLM cleaning suggestions")
            return self.llm_suggestions
        except Exception as e:
            logger.error(f"Error getting LLM cleaning suggestions: {e}")
            return None
        
    def get_llm_cleaning_code(self):
        """Get cleaning code from LLM"""
        logger.info("Requesting cleaning code from LLM")
        try:
            prompt = f"""
            You are a data cleaning expert. Based on the following summary of a dataframe, write a Python script to clean the data. 
                        
            Here's the data summary:
            {json.dumps(self.summary, indent=2)}
                        
            Write a complete Python function called 'clean_data' that takes a dataframe as input and returns a cleaned dataframe.
            The function should:
            1. Handle missing values appropriately
            2. Fix data types and convert columns to their correct types (e.g., convert string numbers to numeric, dates to datetime)
            3. Fix any type errors or inconsistencies within columns
            4. Remove outliers if necessary
            5. Standardize text fields if present
            6. Add useful data transformations
            7. Identify and remove any index columns (columns named 'index', 'id', 'Unnamed: 0', etc. or having sequential integers)
            8. Explicitly convert at least one numeric column to a different type (e.g., float to int or int to float) where appropriate

            IMPORTANT: Include ALL necessary import statements at the top of the script.
            Only use these libraries which are already imported: pandas, numpy, sklearn, scipy.
            DO NOT use or import any other libraries.
            IMPORTANT CONSTRAINTS:
            1. Before any statistical operations (zscore, IQR, etc.), CHECK NUMERIC TYPE:
            if pd.api.types.is_numeric_dtype(df[col]): 
                # perform operation

            2. Use ONLY these available functions/classes:
            {list(self.shared_namespace.keys())}

            3. Handle errors gracefully with try-except blocks

            4. For datetime conversion use:
            df[col] = pd.to_datetime(df[col], errors='coerce')

            5. For text standardization use:
            df[col] = df[col].str.<operations> (no external text libraries)

            6. For numeric type conversion use:
            df[col] = pd.to_numeric(df[col], errors='coerce')

            7. Never use functions not listed above.
            8. Do not use any external libraries or APIs.
                        
            Return ONLY the Python code without any explanations. The code should be production-ready and handle all edge cases.            
            """
            
            response = self.llm.invoke(prompt)
            self.cleaning_code = extract_python_code(response.content)
            
            # Save the cleaning script to a file
            with open("llm_clean_data_script.py", "w") as f:
                f.write(self.cleaning_code)
                
            logger.info("Successfully received and saved LLM cleaning code")
            return self.cleaning_code
        except Exception as e:
            logger.error(f"Error getting LLM cleaning code: {e}")
            return None
            
    def execute_llm_cleaning_code(self):
        """Execute the cleaning code from LLM"""
        logger.info("Executing LLM cleaning code")
        try:
            # Create a shared namespace with necessary imports
            self.shared_namespace = {
                # Basic data science libraries
                "pd": pd,
                "np": np,
                "SimpleImputer": SimpleImputer,
                "stats": stats,
                "StandardScaler": StandardScaler,
                "LabelEncoder": LabelEncoder,
                "IsolationForest": IsolationForest,
                
                # Additional transformers
                "FunctionTransformer": FunctionTransformer,
                "KBinsDiscretizer": KBinsDiscretizer,
                "OrdinalEncoder": OrdinalEncoder,
                "IterativeImputer": IterativeImputer,
                "zscore": zscore,
                "iqr": iqr,
                "date_parser": date_parser,
                "re": re,
                "ast": ast,
                "unicodedata": unicodedata,
                "is_numeric_dtype": pd.api.types.is_numeric_dtype,
                "is_string_dtype": pd.api.types.is_string_dtype,
                "to_numeric": pd.to_numeric,
                "to_datetime": pd.to_datetime,
                "errors": pd.errors,
                "cut": pd.cut,
                "qcut": pd.qcut,
            }
            
            # Execute the cleaning code
            exec(self.cleaning_code, self.shared_namespace, self.shared_namespace)
            
            # Call the clean_data function if it exists
            if "clean_data" in self.shared_namespace:
                self.cleaned_df = self.shared_namespace["clean_data"](self.df)
                logger.info("Successfully executed LLM cleaning code")
                return self.cleaned_df
            else:
                logger.error("The 'clean_data' function was not found in the generated code")
                return None
        except Exception as e:
            logger.error(f"Error executing LLM cleaning code: {e}")
            return None
            
    def run(self):
        """Run the full pipeline"""
        logger.info("Starting LLM Data Cleaning Pipeline")
        
        # Step 1: Load data
        if not self.load_data():
            return False
            
        # Step 2: Generate summary
        self.generate_summary()
            
        # Step 3: Detect issues
        self.detect_issues()
        
        # Step 4: Initialize LLM
        if not self.initialize_llm():
            logger.warning("Could not initialize LLM")
            return False
            
        # Step 5: Get LLM cleaning suggestions
        suggestions = self.get_llm_cleaning_suggestions()
        if not suggestions:
            return False
            
        # Step 6: Get LLM cleaning code
        cleaning_code = self.get_llm_cleaning_code()
        if not cleaning_code:
            return False
        
        # Step 7: Execute LLM cleaning code
        cleaned_df = self.execute_llm_cleaning_code()
        if cleaned_df is None:
            return False
        
        return {
            "cleaned_df": self.cleaned_df,
            "suggestions": self.llm_suggestions,
            "cleaning_code": self.cleaning_code
        }