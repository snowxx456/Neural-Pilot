import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats

def clean_data(df):
    df = df.copy()
    
    try:
        # Handle missing values
        df['PRCP'] = df['PRCP'].fillna(0)
        df['RAIN'] = df['RAIN'].fillna('True')
    except Exception as e:
        print(f"Error handling missing values: {e}")
    
    try:
        # Convert DATE to datetime
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    except Exception as e:
        print(f"Error converting DATE: {e}")
    
    try:
        # Convert PRCP to numeric
        df['PRCP'] = pd.to_numeric(df['PRCP'], errors='coerce')
    except Exception as e:
        print(f"Error converting PRCP: {e}")
    
    try:
        # Convert TMAX and TMIN to integer types
        df['TMAX'] = pd.to_numeric(df['TMAX'], errors='coerce').astype(int)
        df['TMIN'] = pd.to_numeric(df['TMIN'], errors='coerce').astype(int)
    except Exception as e:
        print(f"Error converting TMAX/TMIN: {e}")
    
    try:
        # Convert RAIN to boolean
        df['RAIN'] = df['RAIN'].apply(lambda x: x.lower() == 'true' if isinstance(x, str) else x)
        df['RAIN'] = df['RAIN'].astype(bool)
    except Exception as e:
        print(f"Error converting RAIN: {e}")
    
    try:
        # Remove any index columns
        index_cols = ['index', 'id', 'Unnamed: 0']
        for col in index_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
    except Exception as e:
        print(f"Error removing index columns: {e}")
    
    try:
        # Handle outliers using IQR for numeric columns
        for col in ['PRCP', 'TMAX', 'TMIN']:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if col == 'PRCP':
                    lower_bound = max(lower_bound, 0)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    except Exception as e:
        print(f"Error handling outliers: {e}")
    
    return df