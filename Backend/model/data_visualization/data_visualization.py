import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from typing import Dict, Any
import json
import hashlib

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_SIZE = 10000  # Define a sample size for subsampling large datasets

# Cache for storing datasets in memory (in production, use a proper cache)
dataset_cache = {}

def compute_df_hash(df: pd.DataFrame) -> str:
    """Optimized dataframe hashing"""
    return hashlib.md5(
        pd.util.hash_pandas_object(df.iloc[:min(100, len(df))]).values
    ).hexdigest()

def is_potential_date_column(series: pd.Series, sample_size: int = 5) -> bool:
    """Check if column might contain dates"""
    # Check column name first
    if any(keyword in series.name.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
        return True
    
    # Check sample values
    sample = series.dropna().head(sample_size).astype(str)
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',      # MM/DD/YYYY
        r'\d{2}-\w{3}-\d{2,4}',    # DD-MON-YY(Y)
        r'\d{1,2} \w{3,} \d{4}'    # 1 January 2023
    ]
    
    date_count = sum(1 for val in sample if any(re.match(p, val) for p in date_patterns))
    return date_count / len(sample) > 0.5 if len(sample) > 0 else False

def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column types efficiently."""
    column_types = {}
    
    for column in df.columns:
        # Check for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            # Detect if it's a binary column (0/1, True/False)
            if df[column].nunique() <= 2:
                column_types[column] = "BINARY"
            # Detect if it's a discrete numeric column (few unique values)
            elif df[column].nunique() < 20:
                column_types[column] = "NUMERIC_DISCRETE"
            # Otherwise it's a continuous numeric column
            else:
                column_types[column] = "NUMERIC_CONTINUOUS"
        else:
            # Check for temporal/date columns
            if is_potential_date_column(df[column]):
                try:
                    # Attempt conversion with coerce
                    converted = pd.to_datetime(df[column], errors='coerce')
                    if not converted.isnull().all():  # At least some valid dates
                        column_types[column] = "TEMPORAL"
                        continue
                except Exception:
                    pass
            
            # Check for ID-like columns (high cardinality with unique patterns)
            if (df[column].nunique() > len(df) * 0.9 and 
                any(x in column.lower() for x in ['id', 'code', 'key', 'uuid', 'identifier'])):
                column_types[column] = "ID"
            # Check for categorical columns (low to medium cardinality)
            elif df[column].nunique() <= 20:
                column_types[column] = "CATEGORICAL"
            # Otherwise it's a text column
            else:
                column_types[column] = "TEXT"
   
    return column_types

def get_corr_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute the correlation matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # If we have too many numeric columns, sample them for better performance
    if len(numeric_cols) > 30:
        numeric_cols = numeric_cols[:30]
    
    # Return correlation matrix if we have at least 2 numeric columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        return {
            "columns": corr.columns.tolist(),
            "values": corr.values.tolist()
        }
    return None

def get_subsampled_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return subsampled data for faster visualization."""
    if column not in df.columns:
        return pd.DataFrame()
    
    # Use stratified sampling for categorical columns if possible
    if df[column].nunique() < 20 and len(df) > SAMPLE_SIZE:
        try:
            fractions = min(0.5, SAMPLE_SIZE / len(df))
            return df[[column]].groupby(column, group_keys=False).apply(
                lambda x: x.sample(max(1, int(fractions * len(x))), random_state=42)
            )
        except Exception:
            pass
    
    # Use random sampling
    return df[[column]].sample(min(len(df), SAMPLE_SIZE), random_state=42)

def create_chart(df: pd.DataFrame, column: str, column_type: str) -> Dict[str, Any]:
    """Generate optimized charts based on column type."""
    if column not in df.columns:
        return None
        
    df_sample = get_subsampled_data(df, column)
    if df_sample.empty:
        return None
    
    try:
        fig = None
        
        # Binary columns (0/1, True/False)
        if column_type == "BINARY":
            value_counts = df_sample[column].value_counts()
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Distribution", "Percentage"),
                               specs=[[{"type": "bar"}, {"type": "pie"}]], 
                               column_widths=[0.5, 0.5])
            
            fig.add_trace(go.Bar(
                x=value_counts.index, 
                y=value_counts.values, 
                marker_color=['#FF4B4B', '#4CAF50'],
                text=value_counts.values,
                textposition='auto'
            ), row=1, col=1)
            
            fig.add_trace(go.Pie(
                labels=value_counts.index, 
                values=value_counts.values,
                marker=dict(colors=['#FF4B4B', '#4CAF50']),
                textinfo='percent+label'
            ), row=1, col=2)
            
            fig.update_layout(title_text=f"Binary Distribution: {column}")
        
        # Numeric continuous columns
        elif column_type == "NUMERIC_CONTINUOUS":
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=("Distribution", "Box Plot", "Violin Plot", "Cumulative Distribution"),
                               specs=[[{"type": "histogram"}, {"type": "box"}], 
                                      [{"type": "violin"}, {"type": "scatter"}]])
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=df_sample[column], 
                nbinsx=30, 
                marker_color='#FF4B4B',
                opacity=0.7
            ), row=1, col=1)
            
            # Box plot
            fig.add_trace(go.Box(
                x=df_sample[column], 
                marker_color='#FF4B4B',
                boxpoints='outliers'
            ), row=1, col=2)
            
            # Violin plot
            fig.add_trace(go.Violin(
                x=df_sample[column], 
                marker_color='#FF4B4B',
                box_visible=True,
                points='outliers'
            ), row=2, col=1)
            
            # CDF
            sorted_data = np.sort(df_sample[column].dropna())
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            fig.add_trace(go.Scatter(
                x=sorted_data,
                y=cumulative,
                mode='lines',
                line=dict(color='#FF4B4B', width=2)
            ), row=2, col=2)
            
            fig.update_layout(height=600, title_text=f"Continuous Variable Analysis: {column}")
        
        # Numeric discrete columns
        elif column_type == "NUMERIC_DISCRETE":
            value_counts = df_sample[column].value_counts().sort_index()
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Distribution", "Percentage"),
                               specs=[[{"type": "bar"}, {"type": "pie"}]])
            
            fig.add_trace(go.Bar(
                x=value_counts.index, 
                y=value_counts.values, 
                marker_color='#FF4B4B',
                text=value_counts.values,
                textposition='auto'
            ), row=1, col=1)
            
            fig.add_trace(go.Pie(
                labels=value_counts.index, 
                values=value_counts.values,
                marker=dict(colors=px.colors.sequential.Reds),
                textinfo='percent+label'
            ), row=1, col=2)
            
            fig.update_layout(title_text=f"Discrete Numeric Distribution: {column}")
        
        # Categorical columns
        elif column_type == "CATEGORICAL":
            value_counts = df_sample[column].value_counts().head(20)
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Category Distribution", "Percentage Breakdown"),
                               specs=[[{"type": "bar"}, {"type": "pie"}]])
            
            fig.add_trace(go.Bar(
                x=value_counts.index, 
                y=value_counts.values, 
                marker_color='#00FFA3',
                text=value_counts.values,
                textposition='auto'
            ), row=1, col=1)
            
            fig.add_trace(go.Pie(
                labels=value_counts.index, 
                values=value_counts.values,
                marker=dict(colors=px.colors.sequential.Greens),
                textinfo='percent+label'
            ), row=1, col=2)
            
            fig.update_layout(title_text=f"Categorical Analysis: {column}")
        
        # Temporal/date columns
        elif column_type == "TEMPORAL":
            dates = pd.to_datetime(df_sample[column], errors='coerce', format='mixed')
            valid_dates = dates[dates.notna()]
            
            fig = make_subplots(
                rows=2, 
                cols=2,
                subplot_titles=("Monthly Pattern", "Yearly Pattern", "Cumulative Trend", "Day of Week Distribution"),
                specs=[[{"type": "bar"}, {"type": "bar"}], 
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            if not valid_dates.empty:
                monthly_counts = valid_dates.dt.month.value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_labels = [month_names[i-1] for i in monthly_counts.index]
                
                fig.add_trace(go.Bar(
                    x=month_labels,
                    y=monthly_counts.values,
                    marker_color='#7B68EE',
                    text=monthly_counts.values,
                    textposition='auto'
                ), row=1, col=1)
                
                yearly_counts = valid_dates.dt.year.value_counts().sort_index()
                
                fig.add_trace(go.Bar(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    marker_color='#7B68EE',
                    text=yearly_counts.values,
                    textposition='auto'
                ), row=1, col=2)
                
                sorted_dates = valid_dates.sort_values()
                cumulative = np.arange(1, len(sorted_dates) + 1)
                
                fig.add_trace(go.Scatter(
                    x=sorted_dates,
                    y=cumulative,
                    mode='lines',
                    line=dict(color='#7B68EE', width=2)
                ), row=2, col=1)
                
                dow_counts = valid_dates.dt.dayofweek.value_counts().sort_index()
                dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                dow_labels = [dow_names[i] for i in dow_counts.index]
                
                fig.add_trace(go.Bar(
                    x=dow_labels,
                    y=dow_counts.values,
                    marker_color='#7B68EE',
                    text=dow_counts.values,
                    textposition='auto'
                ), row=2, col=2)
            
            fig.update_layout(height=600, title_text=f"Temporal Analysis: {column}")
        
        # ID columns
        elif column_type == "ID":
            id_lengths = df_sample[column].astype(str).str.len()
            id_prefixes = df_sample[column].astype(str).str[:2].value_counts().head(15)
            
            fig = make_subplots(
                rows=1, 
                cols=2,
                subplot_titles=("ID Length Distribution", "Common ID Prefixes"),
                specs=[[{"type": "histogram"}, {"type": "bar"}]]
            )
            
            fig.add_trace(go.Histogram(
                x=id_lengths,
                nbinsx=20,
                marker_color='#9C27B0'
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=id_prefixes.index,
                y=id_prefixes.values,
                marker_color='#9C27B0',
                text=id_prefixes.values,
                textposition='auto'
            ), row=1, col=2)
            
            fig.update_layout(title_text=f"ID Analysis: {column}")
        
        # Text columns
        elif column_type == "TEXT":
            value_counts = df_sample[column].value_counts().head(15)
            text_lengths = df_sample[column].astype(str).str.len()
            
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=("Top Values", "Text Length Distribution"),
                specs=[[{"type": "bar"}], [{"type": "histogram"}]]
            )
            
            fig.add_trace(
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color='#00B4D8',
                    text=value_counts.values,
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(
                    x=text_lengths,
                    nbinsx=20,
                    marker_color='#00B4D8'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                title_text=f"Text Analysis: {column}"
            )
        
        # Fallback for any other column type
        else:
            fig = go.Figure(go.Histogram(x=df_sample[column], marker_color='#888'))
            fig.update_layout(title_text=f"Generic Analysis: {column}")

        # Common layout settings
        fig.update_layout(
            height=400, 
            showlegend=False, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#FFFFFF'),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return json.loads(fig.to_json())
    
    except Exception as e:
        print(f"Error creating chart for {column}: {str(e)}")
        return None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and return basic info"""
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(contents))
        elif file.filename.endswith('.json'):
            df = pd.read_json(BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Compute basic info
        df_hash = compute_df_hash(df)
        column_types = get_column_types(df)
        corr_matrix = get_corr_matrix(df)
        
        # Store in cache
        dataset_cache[df_hash] = {
            "df": df,
            "column_types": column_types,
            "corr_matrix": corr_matrix,
            "filename": file.filename
        }
        
        return {
            "status": "success",
            "filename": file.filename,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": list(df.columns),
            "column_types": column_types,
            "df_hash": df_hash
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/visualize")
async def visualize_data(df_hash: str, columns: list[str]):
    """Generate visualizations for selected columns"""
    if df_hash not in dataset_cache:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    data = dataset_cache[df_hash]
    df = data["df"]
    column_types = data["column_types"]
    corr_matrix = data["corr_matrix"]
    
    visualizations = {}
    
    for column in columns:
        if column in column_types:
            fig = create_chart(df, column, column_types[column])
            if fig:
                visualizations[column] = fig
    
    # Add correlation matrix if available
    if corr_matrix:
        corr_fig = px.imshow(corr_matrix["values"], 
                            x=corr_matrix["columns"], 
                            y=corr_matrix["columns"], 
                            color_continuous_scale="RdBu")
        visualizations["correlation_matrix"] = json.loads(corr_fig.to_json())
    
    return {
        "status": "success",
        "visualizations": visualizations
    }

@app.get("/sample_data")
async def get_sample_data(df_hash: str, column: str):
    """Get sample data for a column"""
    if df_hash not in dataset_cache:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataset_cache[df_hash]["df"]
    
    if column not in df.columns:
        raise HTTPException(status_code=404, detail="Column not found")
    
    sample = df[column].head(100)
    
    if pd.api.types.is_numeric_dtype(sample):
        stats = df[column].describe().to_dict()
    else:
        stats = df[column].value_counts().to_dict()
    
    return {
        "status": "success",
        "sample": sample.tolist(),
        "stats": stats
    }