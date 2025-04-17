import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats

def clean_data(df):
    try:
        # Handle missing values
        df = df.dropna(how='all')
    except Exception as e:
        print(f"Error handling missing values: {e}")

    try:
        # Convert data types
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Error converting data types: {e}")

    try:
        # Remove outliers
        columns_to_check = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
        mask = pd.Series(True, index=df.index)
        for col in columns_to_check:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                mask &= col_mask
        df = df[mask]
    except Exception as e:
        print(f"Error removing outliers: {e}")

    try:
        # Explicitly convert GradeClass to integer
        df['GradeClass'] = pd.to_numeric(df['GradeClass'], errors='coerce', downcast='integer')
    except Exception as e:
        print(f"Error converting GradeClass: {e}")

    try:
        # Remove index columns
        index_columns = ['index', 'id']
        for col in index_columns:
            if col in df.columns:
                df = df.drop(col, axis=1)
    except Exception as e:
        print(f"Error removing index columns: {e}")

    return df