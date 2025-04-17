import pandas as pd
import numpy as np
from sklearn import *
from scipy import *

def clean_data(df):
    try:
        # Handle missing values
        if df.isnull().values.any():
            df.dropna(inplace=True)
        
        # Convert date column to datetime
        date_col = [col for col in df.columns if 'date' in lower(col)]
        for col in date_col:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns to appropriate types
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert volume to integer type
        if 'volume' in lower(df.columns):
            vol_col = [col for col in df.columns if 'volume' in lower(col)][0]
            df[vol_col] = df[vol_col].astype(np.int64)
        
        # Remove index columns
        index_cols = [col for col in df.columns if col.lower() in ['index', 'id', 'unnamed']]
        df.drop(index_cols, axis=1, inplace=True)
        
        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        return df
    
    except Exception as e:
        print(f"Error in data cleaning: {str(e)}")
        return df