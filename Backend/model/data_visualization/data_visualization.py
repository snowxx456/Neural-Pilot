import re
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Create output directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

SAMPLE_SIZE = 10000  # Define a sample size for subsampling large datasets

def load_data(file_path):
    """Load data from file path"""
    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(file_path)
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
        elif file_extension == "json":
            df = pd.read_json(file_path)
        elif file_extension == "parquet":
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

# Efficiently hash a dataframe to detect changes
def compute_df_hash(df):
    """Optimized dataframe hashing"""
    return hash((df.shape, pd.util.hash_pandas_object(df.iloc[:min(100, len(df))]).sum()))  # Sample-based hashing

def is_potential_date_column(series, sample_size=5):
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
    return date_count / len(sample) > 0.5 if len(sample) > 0 else False  # >50% match

def get_column_types(df):
    """Detect column types efficiently."""
    column_types = {}
    
    # Process columns in batches for better performance
    for chunk_start in range(0, len(df.columns), 10):
        chunk_end = min(chunk_start + 10, len(df.columns))
        chunk_columns = df.columns[chunk_start:chunk_end]
        
        for column in chunk_columns:
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

def get_corr_matrix(df):
    """Compute the correlation matrix for numeric columns."""
    # Only select numeric columns to avoid errors
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # If we have too many numeric columns, sample them for better performance
    if len(numeric_cols) > 30:
        numeric_cols = numeric_cols[:30]
    
    # Return correlation matrix if we have at least 2 numeric columns
    return df[numeric_cols].corr() if len(numeric_cols) > 1 else None

def get_subsampled_data(df, column):
    """Return subsampled data for faster visualization."""
    # Check if column exists
    if column not in df.columns:
        return pd.DataFrame()
    
    # Use stratified sampling for categorical columns if possible
    if df[column].nunique() < 20 and len(df) > SAMPLE_SIZE:
        try:
            # Try to get a representative sample
            fractions = min(0.5, SAMPLE_SIZE / len(df))
            return df[[column]].groupby(column, group_keys=False).apply(
                lambda x: x.sample(max(1, int(fractions * len(x))), random_state=42)
            )
        except Exception:
            # Fall back to random sampling
            pass
    
    # Use random sampling
    return df[[column]].sample(min(len(df), SAMPLE_SIZE), random_state=42)

def save_figure(fig, filename):
    """Save figure to visualizations folder"""
    filepath = os.path.join('visualizations', filename)
    fig.write_html(filepath)
    print(f"Saved visualization to {filepath}")

def create_chart(df, column, column_type):
    """Generate optimized charts based on column type."""
    # Check if column exists in the dataframe
    if column not in df.columns:
        return None
        
    # Get subsampled data for better performance
    df_sample = get_subsampled_data(df, column)
    if df_sample.empty:
        return None
    
    try:
        # Year-based columns (special case)
        if "year" in column.lower():
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Year Distribution", "Box Plot"),
                               specs=[[{"type": "bar"}, {"type": "box"}]], column_widths=[0.7, 0.3], horizontal_spacing=0.1)
            year_counts = df_sample[column].value_counts().sort_index()
            fig.add_trace(go.Bar(x=year_counts.index, y=year_counts.values, marker_color='#7B68EE'), row=1, col=1)
            fig.add_trace(go.Box(x=df_sample[column], marker_color='#7B68EE'), row=1, col=2)
            save_figure(fig, f"{column}_year_analysis.html")
        
        # Binary columns (0/1, True/False)
        elif column_type == "BINARY":
            value_counts = df_sample[column].value_counts()
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Distribution", "Percentage"),
                               specs=[[{"type": "bar"}, {"type": "pie"}]], 
                               column_widths=[0.5, 0.5], 
                               horizontal_spacing=0.1)
            
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
            save_figure(fig, f"{column}_binary_analysis.html")
        
        # Numeric continuous columns
        elif column_type == "NUMERIC_CONTINUOUS":
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=("Distribution", "Box Plot", "Violin Plot", "Cumulative Distribution"),
                               specs=[[{"type": "histogram"}, {"type": "box"}], 
                                      [{"type": "violin"}, {"type": "scatter"}]], 
                               vertical_spacing=0.15,
                               horizontal_spacing=0.1)
            
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
            save_figure(fig, f"{column}_continuous_analysis.html")
        
        # Numeric discrete columns
        elif column_type == "NUMERIC_DISCRETE":
            value_counts = df_sample[column].value_counts().sort_index()
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Distribution", "Percentage"),
                               specs=[[{"type": "bar"}, {"type": "pie"}]], 
                               column_widths=[0.7, 0.3], 
                               horizontal_spacing=0.1)
            
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
            save_figure(fig, f"{column}_discrete_analysis.html")
        
        # Categorical columns
        elif column_type == "CATEGORICAL":
            value_counts = df_sample[column].value_counts().head(20)  # Limit to top 20 categories
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Category Distribution", "Percentage Breakdown"),
                               specs=[[{"type": "bar"}, {"type": "pie"}]], 
                               column_widths=[0.6, 0.4], 
                               horizontal_spacing=0.1)
            
            # Bar chart
            fig.add_trace(go.Bar(
                x=value_counts.index, 
                y=value_counts.values, 
                marker_color='#00FFA3',
                text=value_counts.values,
                textposition='auto'
            ), row=1, col=1)
            
            # Pie chart
            fig.add_trace(go.Pie(
                labels=value_counts.index, 
                values=value_counts.values,
                marker=dict(colors=px.colors.sequential.Greens),
                textinfo='percent+label'
            ), row=1, col=2)
            
            fig.update_layout(title_text=f"Categorical Analysis: {column}")
            save_figure(fig, f"{column}_categorical_analysis.html")
        
        # Temporal/date columns
        elif column_type == "TEMPORAL":
            # Convert with safe datetime parsing
            dates = pd.to_datetime(df_sample[column], errors='coerce', format='mixed')
            valid_dates = dates[dates.notna()]
            
            fig = make_subplots(
                rows=2, 
                cols=2,
                subplot_titles=("Monthly Pattern", "Yearly Pattern", "Cumulative Trend", "Day of Week Distribution"),
                vertical_spacing=0.15,
                horizontal_spacing=0.1,
                specs=[[{"type": "bar"}, {"type": "bar"}], 
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Monthly pattern
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
                
                # Yearly pattern
                yearly_counts = valid_dates.dt.year.value_counts().sort_index()
                
                fig.add_trace(go.Bar(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    marker_color='#7B68EE',
                    text=yearly_counts.values,
                    textposition='auto'
                ), row=1, col=2)
                
                # Cumulative trend
                sorted_dates = valid_dates.sort_values()
                cumulative = np.arange(1, len(sorted_dates) + 1)
                
                fig.add_trace(go.Scatter(
                    x=sorted_dates,
                    y=cumulative,
                    mode='lines',
                    line=dict(color='#7B68EE', width=2)
                ), row=2, col=1)
                
                # Day of week distribution
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
            save_figure(fig, f"{column}_temporal_analysis.html")
        
        # ID columns (show distribution of first few characters, length distribution)
        elif column_type == "ID":
            # Calculate ID length statistics
            id_lengths = df_sample[column].astype(str).str.len()
            
            # Extract first 2 characters for prefix analysis
            id_prefixes = df_sample[column].astype(str).str[:2].value_counts().head(15)
            
            fig = make_subplots(
                rows=1, 
                cols=2,
                subplot_titles=("ID Length Distribution", "Common ID Prefixes"),
                horizontal_spacing=0.1,
                specs=[[{"type": "histogram"}, {"type": "bar"}]]
            )
            
            # ID length histogram
            fig.add_trace(go.Histogram(
                x=id_lengths,
                nbinsx=20,
                marker_color='#9C27B0'
            ), row=1, col=1)
            
            # ID prefix bar chart
            fig.add_trace(go.Bar(
                x=id_prefixes.index,
                y=id_prefixes.values,
                marker_color='#9C27B0',
                text=id_prefixes.values,
                textposition='auto'
            ), row=1, col=2)
            
            fig.update_layout(title_text=f"ID Analysis: {column}")
            save_figure(fig, f"{column}_id_analysis.html")
        
        # Text columns
        elif column_type == "TEXT":
            # For text columns, show top values and length distribution
            value_counts = df_sample[column].value_counts().head(15)
            
            # Calculate text length statistics
            text_lengths = df_sample[column].astype(str).str.len()
            
            fig = make_subplots(
                rows=2, 
                cols=1,
                subplot_titles=("Top Values", "Text Length Distribution"),
                vertical_spacing=0.2,
                specs=[[{"type": "bar"}], [{"type": "histogram"}]]
            )
            
            # Top values bar chart
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
            
            # Text length histogram
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
            save_figure(fig, f"{column}_text_analysis.html")
        
        # Fallback for any other column type
        else:
            fig = go.Figure(go.Histogram(x=df_sample[column], marker_color='#888'))
            fig.update_layout(title_text=f"Generic Analysis: {column}")
            save_figure(fig, f"{column}_generic_analysis.html")

        # Common layout settings
        fig.update_layout(
            height=400, 
            showlegend=False, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#FFFFFF'),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating chart for {column}: {str(e)}")
        return None

def visualize_data(file_path, selected_columns=None):
    """Generate visualizations for the dataset"""
    # Load data
    df = load_data(file_path)
    
    if df is None:
        print("No data available. Please provide a valid dataset.")
        return
    
    print(f"📊 Dataset loaded with shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Calculate dataframe hash
    df_hash = compute_df_hash(df)  

    # If no columns specified, visualize all columns
    if selected_columns is None:
        selected_columns = list(df.columns)
    else:
        # If specific columns are provided, use only those that exist in the dataframe
        selected_columns = [col for col in selected_columns if col in df.columns]
    
    # Get column types and correlation matrix
    print("🔍 Analyzing data structure...")
    column_types = get_column_types(df)
    corr_matrix = get_corr_matrix(df)
    
    if selected_columns:
        print("📈 Generating visualizations for all columns...")
        for column in selected_columns:
            # Only create chart if column exists in column_types
            if column in column_types:
                print(f"  - Processing column: {column} ({column_types[column]})")
                fig = create_chart(df, column, column_types[column])
            else:
                print(f"⚠️ Column '{column}' not found in the dataset or its type couldn't be determined.")

        if corr_matrix is not None:
            print("🔗 Generating correlation matrix...")
            fig = px.imshow(corr_matrix, title="Correlation Matrix", color_continuous_scale="RdBu")
            save_figure(fig, "correlation_matrix.html")
    
    else:
        print("👆 Please specify columns to visualize")

# Main function to run the analysis
def visualization(file_path):
    columns_to_visualize = None  # None means visualize all columns, or specify a list
    
    visualize_data(file_path, columns_to_visualize)
    print("✅ Visualizations saved to 'visualizations' folder")