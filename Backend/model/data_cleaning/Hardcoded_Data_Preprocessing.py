# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import re
import os
import uuid
from typing import Optional
from pathlib import Path

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory to store uploaded and cleaned files
# UPLOAD_DIR = Path("uploads")
# UPLOAD_DIR.mkdir(exist_ok=True)

def load_file(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Remove currency/numeric symbols from string values like $45, ₹1000, etc.
    symbol_pattern = re.compile(r'[^\d\.\-]')  # Keep digits, decimal point, and minus sign

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            def convert_symbol_value(val):
                if pd.isnull(val):
                    return val
                if isinstance(val, str):
                    stripped = re.sub(symbol_pattern, '', val)
                    try:
                        return float(stripped)
                    except ValueError:
                        return val  # Keep original if conversion fails
                return val

            cleaned[col] = cleaned[col].apply(convert_symbol_value)

    # Handle columns based on missingness
    for col in cleaned.columns:
        missing_percentage = cleaned[col].isnull().mean() * 100

        if missing_percentage > 60:
            cleaned.drop(col, axis=1, inplace=True)
        elif missing_percentage > 20:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col].fillna(cleaned[col].mean(), inplace=True)
            elif cleaned[col].dtype == "object":
                mode_series = cleaned[col].mode()
                if not mode_series.empty:
                    cleaned[col].fillna(mode_series.iloc[0], inplace=True)
            else:
                cleaned[col].fillna("unknown", inplace=True)
        else:
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
            cleaned.drop(index, axis=0, inplace=True)
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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and cleaning"""
    try:
        # Generate a unique ID for this session
        session_id = str(uuid.uuid4())
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir()
        
        # Save original file
        original_path = session_dir / file.filename
        with open(original_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the file
        df = load_file(original_path)
        cleaned_df = preprocess(df)
        
        # Save cleaned file
        cleaned_filename = f"cleaned_{file.filename.split('.')[0]}.csv"
        cleaned_path = session_dir / cleaned_filename
        cleaned_df.to_csv(cleaned_path, index=False)
        
        # Get some stats about the cleaning
        original_shape = df.shape
        cleaned_shape = cleaned_df.shape
        
        return {
            "status": "success",
            "session_id": session_id,
            "original_filename": file.filename,
            "cleaned_filename": cleaned_filename,
            "original_shape": {"rows": original_shape[0], "cols": original_shape[1]},
            "cleaned_shape": {"rows": cleaned_shape[0], "cols": cleaned_shape[1]},
            "columns": list(cleaned_df.columns),
            "preview": cleaned_df.head(5).to_dict(orient="records")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download the cleaned file"""
    file_path = UPLOAD_DIR / session_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

@app.get("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session files"""
    session_dir = UPLOAD_DIR / session_id
    if session_dir.exists():
        for file in session_dir.iterdir():
            file.unlink()
        session_dir.rmdir()
    return {"status": "success"}