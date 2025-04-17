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

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

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
        """Generate enhanced data profile with better column type signals"""
        self.total_rows = len(self.df)
        summary = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing = self.df[col].isnull().sum()
            total = len(self.df)
            missing_pct = round((missing / total) * 100, 2)
            unique_count = self.df[col].nunique()
            unique_ratio = round(unique_count / total * 100, 2)

            # Sample values with numpy type conversion
            sample_vals = [str(x) for x in self.df[col].dropna().astype(str).unique()[:10]]

            stats = {
                "name": col,
                "dtype": dtype,
                "contains_id_in_name": "id" in col.lower() or "_id" in col.lower(),
                "missing_percent": float(missing_pct),
                "unique_count": int(unique_count),
                "unique_ratio": float(unique_ratio),
                "sample_values": sample_vals
            }

            if pd.api.types.is_numeric_dtype(self.df[col]):
                values = self.df[col].dropna().values
                if len(values) > 1:
                    sorted_vals = sorted(values[:100]) if len(values) > 100 else sorted(values)
                    is_sequential = False
                    if len(sorted_vals) > 5:
                        diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1)]
                        is_sequential = bool(  # Convert numpy bool to Python bool
                            abs(np.mean(diffs) - 1 < 0.1 and np.std(diffs) < 0.5
                        ))   

                    stats.update({
                        "min": float(self.df[col].min()),
                        "max": float(self.df[col].max()),
                        "mean": float(self.df[col].mean()),
                        "median": float(self.df[col].median()),
                        "std": float(self.df[col].std()),
                        "is_sequential": is_sequential,
                        "has_decimals": any('.' in str(x) for x in values[:100]),
                        "skew": float(self.df[col].skew())
                    })

            # Add classification with type conversion
            stats['classification'] = json.loads(
                json.dumps(
                    self.classify_column(col, self.df[col], stats),
                    cls=NumpyEncoder
                )
            )

            self.summary[col] = stats
        logger.info("Enhanced data summary generated successfully")
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
        """Identify data quality issues with type mismatch detection"""
        issues = {}
        for col, stats in self.summary.items():
            col_issues = []
            
            # Existing issue detection
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

            # New: Classification-based type mismatch detection
            classification = stats.get('classification', {})
            if classification.get('primary_type', 'unknown') != 'unknown':
                dtype_mapping = {
                    'id': 'object',
                    'datetime': 'datetime64',
                    'numeric': ['int64', 'float64'],
                    'categorical': 'category',
                    'text': 'object'
                }
                
                expected_type = dtype_mapping.get(classification['primary_type'], 'unknown')
                actual_dtype = stats['dtype']
                conversion_notes = classification.get('conversion_notes', [])
                
                # Special handling for different types
                if classification['primary_type'] == 'numeric':
                    if actual_dtype not in expected_type:
                        note = " (might need conversion)" if conversion_notes else ""
                        col_issues.append((
                            "critical", 
                            f"Type mismatch: Numeric values stored as {actual_dtype}{note}"
                        ))
                        
                elif classification['primary_type'] == 'datetime':
                    if not actual_dtype.startswith('datetime64'):
                        col_issues.append((
                            "critical", 
                            f"Datetime values stored as {actual_dtype}"
                        ))
                        
                elif classification['primary_type'] == 'id':
                    if actual_dtype != 'object':
                        col_issues.append((
                            "critical", 
                            f"ID column should be string (current: {actual_dtype})"
                        ))
                        
                elif classification['primary_type'] == 'categorical':
                    if actual_dtype != 'category':
                        col_issues.append((
                            "warning", 
                            f"Categorical data stored as {actual_dtype}"
                        ))

                elif classification['primary_type'] == 'text':
                    if actual_dtype != 'object':
                        col_issues.append((
                            "warning", 
                            f"Text data stored as {actual_dtype}"
                        ))

            issues[col] = col_issues
            
        self.issues = issues
        logger.info("Enhanced data issues detected with type validation")
        return issues
    
    def classify_column(self, column_name, column_data, summary_stats):
        """Enhanced column classifier with proper dtype and total_rows usage"""
    
        # Extract key statistics
        sample_values = summary_stats['sample_values']
        dtype = summary_stats['dtype']
        unique_count = summary_stats['unique_count']
        total_rows = self.total_rows
        unique_ratio = unique_count / total_rows
        
        # Initialize classification structure
        result = {
            'primary_type': 'unknown',
            'subtype': None,
            'confidence': 0.0,
            'evidence': [],
            'conversion_notes': [],
            'conflicting_indicators': []
        }

        # Immediate dtype-based returns (converted to string check)
        if dtype.startswith('datetime'):
            return {
                'primary_type': 'datetime',
                'confidence': 1.0,
                'evidence': [f"Inherent dtype: {dtype}"]
            }

        # --------------------
        # 1. ID Column Check (with numpy bool conversion)
        # --------------------
        id_signals = {
            'name_match': any(kw in column_name.lower() 
                            for kw in ['id', 'code', 'key']),
            'high_uniqueness': unique_ratio > 0.95,
            'dtype_mismatch': (
                dtype == 'object' and 
                all(str(x).isdigit() for x in sample_values[:100])
            ),
            'is_sequential': False,
            'uuid_like': False
        }

        if id_signals['name_match'] and pd.api.types.is_numeric_dtype(column_data):
            sorted_sample = sorted([x for x in sample_values if str(x).isdigit()][:100])
            if len(sorted_sample) > 1:
                diffs = [int(sorted_sample[i+1]) - int(sorted_sample[i]) 
                    for i in range(len(sorted_sample)-1)]
                # Convert numpy bool to Python bool
                id_signals['is_sequential'] = bool(
                    (abs(np.mean(diffs) - 1 < 0.1) & 
                    (np.std(diffs) < 0.5)
                ))

        # UUID check
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        id_signals['uuid_like'] = any(re.match(uuid_pattern, str(x)) 
                                    for x in sample_values[:50])

        if sum(id_signals.values()) >= 3:
            result.update({
                'primary_type': 'id',
                'subtype': 'sequential' if id_signals['is_sequential'] else 'identifier',
                'confidence': min(0.95, sum(id_signals.values())*0.3),
                'evidence': [k for k,v in id_signals.items() if v],
                'conversion_notes': ['Convert to string type']
            })
            return result

        # --------------------
        # 2. Datetime Check (enhanced with dtype)
        # --------------------
        if dtype == 'object':
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}',
                r'\d{4}\d{2}\d{2}', r'\d{2}:\d{2}:\d{2}(\.\d+)?'
            ]
            date_matches = sum(
                any(re.search(p, str(x)) for x in sample_values[:50])
                for p in date_patterns
            )
            if 'date' in column_name.lower() or 'time' in column_name.lower():
                date_matches += 2

            if date_matches >= 2:
                result.update({
                    'primary_type': 'datetime',
                    'confidence': min(0.9, date_matches * 0.3),
                    'evidence': [f"Date patterns found ({date_matches})"],
                    'conversion_notes': ['Verify datetime formats']
                })
                return result

        # ----------------------------
        # 3. Special Numeric Check (safe division)
        # ----------------------------
        if len(sample_values) > 0:  # Prevent division by zero
            stripped_numeric = [re.sub(r'[^\d.,-]', '', str(x)) for x in sample_values]
            convertible = sum(1 for sn in stripped_numeric if sn.replace(',', '').replace('.', '').isdigit())
            
            if convertible / len(sample_values) > 0.8:
                result.update({
                    'primary_type': 'numeric',
                    'subtype': 'formatted',
                    'confidence': convertible / len(sample_values),
                    'conversion_notes': [
                        f"Strip non-numeric chars ({convertible/len(sample_values):.0%} convertible)"
                    ],
                    'evidence': [f"Sample conversion rate: {convertible}/{len(sample_values)}"]
                })
                return result

        # --------------------
        # 4. True Numeric Check (dtype-aware)
        # --------------------
        if pd.api.types.is_numeric_dtype(column_data):
            numeric_confidence = 0.9
            if unique_ratio < 0.05:
                numeric_confidence *= 0.7
                result['conflicting_indicators'].append("Low cardinality numeric")
            
            result.update({
                'primary_type': 'numeric',
                'subtype': 'continuous' if summary_stats.get('has_decimals') else 'discrete',
                'confidence': numeric_confidence,
                'evidence': [f"Inherent dtype: {dtype}"]
            })
            return result

        # --------------------
        # 5. Categorical Check (size-aware)
        # --------------------
        cat_threshold = 0.2 if total_rows > 1000 else 0.3
        if unique_ratio < cat_threshold:
            result.update({
                'primary_type': 'categorical',
                'confidence': (1 - unique_ratio) * 0.8,
                'evidence': [
                    f"Unique ratio: {unique_ratio:.1%}",
                    f"Total categories: {unique_count}"
                ]
            })
            return result

        # --------------------
        # 6. Free Text Check (length analysis)
        # --------------------
        avg_length = float(np.mean([len(str(x)) for x in sample_values]))
        result.update({
            'primary_type': 'text',
            'subtype': 'freeform' if avg_length > 25 else 'short_text',
            'confidence': float(min(0.9, avg_length / 50)),  # Explicit float conversion
            'evidence': [
                f"Avg length: {avg_length:.1f} chars",
                f"Unique ratio: {unique_ratio:.1%}"
            ]
        })
        
        return result
    
    def get_llm_column_classification(self):
        """First pass: classify column types with LLM"""
        logger.info("Requesting column classification from LLM")
        try:
            prompt = f"""
                You are a data scientist classifying columns in a dataset for cleaning.
                Given the column information below, classify each column as one of these types:
                - id: unique identifiers, not for numerical analysis
                - categorical: discrete labels or categories, including binary and ordinal
                - numeric: true numerical values meant for calculations
                - datetime: date or time information
                - text: free text fields with sentences or paragraphs
                
                Dataset column information:
                {json.dumps(self.summary, indent=2, cls=NumpyEncoder)}
                
                Return ONLY a JSON dictionary with column names as keys and types as values.
                Example: {{"student_id": "id", "age": "numeric", "grade": "categorical"}}
            """
            
            response = self.llm.invoke(prompt)
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback regex parsing for malformed JSON
                match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0).replace("'", '"'))
                    except json.JSONDecodeError:
                        logger.error("Failed to parse column classifications")
                return {}
        except Exception as e:
            logger.error(f"Column classification failed: {e}")
            return {}
    
    def get_llm_cleaning_suggestions(self):
        """Two-pass approach for cleaning suggestions"""
        column_types = self.get_llm_column_classification()
        
        if not column_types:
            logger.warning("Falling back to single-pass approach")
            return self._legacy_cleaning_suggestions()
            
        logger.info("Requesting type-aware cleaning suggestions")
        try:
            combined_data = {
                "summary": json.loads(json.dumps(self.summary, cls=NumpyEncoder)),
                "issues": self.issues,
                "column_types": column_types
            }
            
            prompt = f"""
                Based on the dataset information and pre-classified column types below,
                provide detailed cleaning recommendations:

                DATASET INFORMATION:
                {json.dumps(combined_data, indent=2)}

                For each column, provide specific cleaning operations based on its type:
                - id: Standardize format, prevent numeric treatment
                - categorical: Handle missing values, standardize categories
                - numeric: Handle outliers, fix scaling issues
                - datetime: Standardize format, handle timezones
                - text: Normalize text, handle encoding

                Required structure:
                {{
                    "columns": {{
                        "column_name": {{
                            "confirmed_type": "type",
                            "anomalies": ["specific issues"],
                            "operations": ["cleaning steps"],
                            "example_fix": "example transformation"
                        }}
                    }},
                    "global_issues": ["cross-column problems"],
                    "priority_order": ["cleaning sequence"]
                }}
                """
            
            response = self.llm.invoke(prompt)
            self.llm_suggestions = response.content
            logger.info("Received type-aware suggestions")
            return self.llm_suggestions
        except Exception as e:
            logger.error(f"Cleaning suggestions failed: {e}")
            return None

    def _legacy_cleaning_suggestions(self):
        """Fallback single-pass approach"""
        logger.info("Using legacy single-pass suggestion method")
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