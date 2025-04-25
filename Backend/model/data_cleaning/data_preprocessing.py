import pandas as pd
import numpy as np
import re
import os

def clean_currency_symbols(df, currency_columns):
    """Clean specified currency columns while preserving other data"""
    for col in currency_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(r'[^\d.,]', '', regex=True)
            df[col] = df[col].str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df  

def load_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

def preprocess(df, currency_columns=[]):
    cleaned = df.copy()

    # Clean currency columns before handling missing values
    cleaned = clean_currency_symbols(cleaned, currency_columns)

    # Handle columns based on missingness
    for col in cleaned.columns:
        missing_percentage = cleaned[col].isnull().mean() * 100
        
        if missing_percentage > 60:
            cleaned.drop(col, axis=1, inplace=True)  # Drop columns with >60% missing
        elif missing_percentage > 20:
            # Impute or flag for 20–60% missing columns
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col].fillna(cleaned[col].mean(), inplace=True)
            elif cleaned[col].dtype == "object":
                mode_series = cleaned[col].mode()
                if not mode_series.empty:
                    cleaned[col].fillna(mode_series.iloc[0], inplace=True)
            else:
                cleaned[col].fillna("unknown", inplace=True)
        else:
            # Impute columns with <20% missing data
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col].fillna(cleaned[col].mean(), inplace=True)
            elif cleaned[col].dtype == "object":
                mode_series = cleaned[col].mode()
                if not mode_series.empty:
                    cleaned[col].fillna(mode_series.iloc[0], inplace=True)
            else:
                cleaned[col].fillna("unknown", inplace=True)

    # Handle rows based on missingness
    for index, row in cleaned.iterrows():
        missing_percentage = row.isnull().mean() * 100
        
        if missing_percentage > 50:
            cleaned.drop(index, axis=0, inplace=True)  # Remove rows with >50% missing
        elif missing_percentage > 20:
            for col in row.index:
                if pd.isnull(row[col]):
                    if pd.api.types.is_numeric_dtype(cleaned[col]):
                        cleaned.at[index, col] = cleaned[col].mean()
                    elif cleaned[col].dtype == "object":
                        mode_series = cleaned[col].mode()
                        if not mode_series.empty:
                            cleaned.at[index, col] = mode_series.iloc[0]
        else:
            for col in row.index:
                if pd.isnull(row[col]):
                    if pd.api.types.is_numeric_dtype(cleaned[col]):
                        cleaned.at[index, col] = cleaned[col].mean()
                    elif cleaned[col].dtype == "object":
                        mode_series = cleaned[col].mode()
                        if not mode_series.empty:
                            cleaned.at[index, col] = mode_series.iloc[0]
                    else:
                        cleaned.at[index, col] = "unknown"

    cleaned.drop_duplicates(inplace=True)
    return cleaned

def main(file_path):
    """Main function to load, preprocess, and save the cleaned data."""

    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    try:
        df = load_file(file_path)
        print("\n✅ Loaded data. Preview:")
        print(df.head())

        # Define currency columns to clean here (manually or auto-detect)
        currency_columns = ['Price', 'Cost']  # <-- replace with actual column names

        cleaned_df = preprocess(df, currency_columns)
        cleaned_df.to_csv("cleaned_data.csv", index=False)

        print("\n🎉 Data cleaned successfully!")
        print("📁 Cleaned file saved as: cleaned_data.csv")
    except Exception as e:
        print(f"❌ Error: {e}")