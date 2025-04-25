from model.data_cleaning.llm.config import Config, set_production
import pandas as pd
from api.models import Dataset
import os
import time
verbose = set_production()

class CreateAgent:
    def __init__(self, data):
        self.data = data
        self.config = Config(self.data)
        self.agent = self.config.get_agent()
    
    def update_agent(self,dataframe):
        """
        Update the agent with the new dataframe.
        """
        self.data = dataframe
        self.config = Config(self.data)
        self.agent = self.config.get_agent()
    
    def find_missing_values(self):
        """
        Find missing values in the dataset.
        :return: Response from the agent about columns with missing values and their counts.
        """
        return self.agent.chat("List the columns with missing values and the count, without showing the entire dataset.")
    
    def handle_index_column(self):
        """
        Find the index column in the dataset.
        Removes the index column if it exists.
        """
        index = self.agent.chat("Which column is being used as the id? if if that exist remove it from the dataset.")
        if isinstance(index, pd.DataFrame):
            self.update_agent(index)
        
    
    def find_target_column(self):
        """
        Find the target column in the dataset.
        :return: Response from the agent about the target column.
        """
        return self.agent.chat(
            """Analyze this dataset and determine which column is most likely to be the target/label for prediction.

            Consider these factors:
            1. Look for columns that could represent outcomes (e.g., 'PRICE', 'SALES', 'RISK', 'DEFAULT', 'APPROVED', 'SUCCESS')
            2. Check for binary/categorical columns that might indicate classification targets (e.g., 'RESULT', 'OUTCOME', 'STATUS')
            3. Examine continuous variables at the end of the dataframe that might be dependent variables
            4. Consider column names containing words like 'TARGET', 'LABEL', 'OUTCOME', 'RESULT', 'PREDICTION'
            5. Check if any column appears to be derived from other columns in the dataset

            Look at the column data types, distributions, and relationships to other columns.
            
            Respond with ONLY the name of the single most likely target column as a string (e.g., "INCOME").
           
            
            - DO NOT use pd.api or any advanced pandas methods
            - DO NOT use try/except blocks or complex error handling
            - DO NOT use inplace=True parameter anywhere
            - DO NOT use df.astype() with parameters that might be restricted
            - Use only the most basic pandas operations
            - Preserve all missing values exactly as they were
            - Use direct assignment for all changes: df['COL'] = df['COL'].operation()
            Do not include any explanations or additional text in your response.
            """
        )
    def handle_data_types(self):
        """
        Intelligently convert data types for all columns while preserving missing values.
        Works with any dataset structure.
        """
        # Store original missing value information
        original_df = self.agent.dataframe.execute().copy()
        missing_mask = original_df.isna()

        if not verbose:
            print("Missing values before conversion:")
            print(original_df.isna().sum())
        
        # Use PandasAI to convert data types with generic rules
        converted_df = self.agent.chat(
                """Analyze the dataset and intelligently convert each column to its appropriate data type using only simple pandas operations:

                DETECTION PHASE - Examine each column carefully:
                1. For each column, determine its likely data type by checking:
                - Column name (keywords like 'ID', 'CODE', 'ZIP', 'POSTAL', 'DATE', 'TIME')
                - Value patterns (digits only, decimal points, date formats, few unique values)
                - Special formatting (leading zeros, dashes, special characters)

                CONVERSION PHASE - Apply these strict rules:
                1. For identifier columns (must keep as strings):
                - ANY column containing: 'ID', 'CODE', 'ZIP', 'POSTAL', 'SSN', 'PHONE', 'LICENSE', 'ACCOUNT'
                - ANY column that appears to be a code system (mixed letters/numbers or formatted numbers)
                - ANY column with leading zeros
                - Example: df['POSTAL_CODE'] = df['POSTAL_CODE'].astype(str)

                2. For categorical columns:
                - Columns with text values or few unique values (<15% of total rows)
                - Human attributes (SEX, GENDER, RACE, etc.)
                - Status fields (STATUS, STATE, CONDITION, etc.)
                - Example: df['MARITAL_STATUS'] = df['MARITAL_STATUS'].astype('category')

                3. For numeric columns:
                - Money/financial values as float64: 'INCOME', 'PRICE', 'COST', 'SALARY', etc.
                - Measurements as float64: 'HEIGHT', 'WEIGHT', 'DISTANCE', 'TIME', etc.
                - Counts that might have missing values as float64
                - Example: df['INCOME'] = df['INCOME'].astype('float64')

                4. For date/time columns:
                - Check if columns with 'DATE', 'TIME', 'YEAR', 'MONTH', 'DAY' can be converted
                - Only convert if format is clearly a date/time
                - Example: df['BIRTH_DATE'] = pd.to_datetime(df['BIRTH_DATE'], errors='ignore')
                - If errors='ignore' is not allowed, simply skip problematic date columns

                VALIDATION PHASE - After conversion:
                1. For each converted column, check if any non-missing values were accidentally lost
                2. If values were lost, revert that column to its original type
                3. Skip any conversion that causes errors rather than attempting complex fixes

                IMPORTANT RESTRICTIONS:
                - DO NOT use pd.api or any advanced pandas methods
                - DO NOT use try/except blocks or complex error handling
                - DO NOT use inplace=True parameter anywhere
                - DO NOT use df.astype() with parameters that might be restricted
                - Use only the most basic pandas operations
                - Preserve all missing values exactly as they were
                - Use direct assignment for all changes: df['COL'] = df['COL'].operation()

                Return the DataFrame with proper types applied and all original missing values preserved.
                """
            )
        
        # If we received a dataframe back, restore any missing values that might have been altered
        if isinstance(converted_df, pd.DataFrame):
            # Apply the missing mask to restore ALL original missing values
            for col in original_df.columns:
                if col in converted_df.columns:
                    # Restore original NaN values
                    converted_df.loc[missing_mask[col], col] = None
            
            if not verbose:
                print("\nMissing values after conversion:")
                print(converted_df.isna().sum())
            
            
            # Update the agent with the fixed dataframe
            self.update_agent(converted_df)

    def handle_outliers(self):
        """
        Directly handle outliers in the dataset using IQR method without relying on the agent.
        """
        # Get the original dataframe
        df = self.agent.dataframe.execute().copy()
        
        # Store original for comparison
        original_df = df.copy()
        
        # Process numerical columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        total_outliers = 0
        
        for col in numerical_cols:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers_lower = df[col] < lower_bound
            outliers_upper = df[col] > upper_bound
            outlier_count = outliers_lower.sum() + outliers_upper.sum()
            total_outliers += outlier_count
            
            print(f"Column {col}: Found {outlier_count} outliers")
            
            # Handle outliers (winsorization)
            df.loc[outliers_lower, col] = lower_bound
            df.loc[outliers_upper, col] = upper_bound
        
        print(f"\nTotal outliers handled: {total_outliers}")
        
        # Print statistics
        print("\nOutlier Handling Summary:")
        for col in numerical_cols:
            print(f"\nColumn: {col}")
            print(f"Before - Min: {original_df[col].min()}, Max: {original_df[col].max()}, Mean: {original_df[col].mean():.2f}, Std: {original_df[col].std():.2f}")
            print(f"After  - Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
        
        # Update the agent with the modified dataframe
        self.update_agent(df)

    def handle_duplicates(self):
        """
        Handle duplicate rows in the dataset.
        """
        # Get the original dataframe
        df = self.agent.dataframe.execute().copy()
        
        # Store original for comparison
        original_df = df.copy()
        
        # Identify duplicates
        duplicate_rows = df.duplicated(keep='first')
        duplicate_count = duplicate_rows.sum()
        
        print(f"Found {duplicate_count} duplicate rows")
        
        # Remove duplicates
        df = df[~duplicate_rows]
        
        # Print statistics
        print("\nDuplicate Handling Summary:")
        print(f"Before - Total Rows: {original_df.shape[0]}")
        print(f"After  - Total Rows: {df.shape[0]}")
        
        # Update the agent with the modified dataframe
        self.update_agent(df)       
        
    def handle_missing_values(self, strategy='auto',max_try=3):
        """More reliable implementation"""
        # Access the actual pandas DataFrame 
        original = self.agent.dataframe.execute().copy()


        for i in range(max_try):
            if self.agent.dataframe.execute().isna().sum().sum() == 0:
                break
            if strategy == 'auto':
                # Step 1: Drop high-missing columns
                result_df = self.agent.chat(
                    "Drop columns where more than 30% of values are missing "
                    "and return the modified dataframe"
                    """
                    - DO NOT use pd.api or any advanced pandas methods
                    - DO NOT use try/except blocks or complex error handling
                    - DO NOT use inplace=True parameter anywhere
                    - DO NOT use df.astype() with parameters that might be restricted
                    - Use only the most basic pandas operations
                    - Preserve all missing values exactly as they were
                    - Use direct assignment for all changes: df['COL'] = df['COL'].operation()"""
                )
                
                # Update the agent with the modified dataframe
                # Assuming the result is a dataframe
                if isinstance(result_df, pd.DataFrame):
                    # We need to update the agent's dataframe with this result
                    # This depends on how your agent is structured
                    self.config = Config(self.data)
                    self.agent = self.config.get_agent()
                    
                # Step 2: Handle remaining missing values
                final_df = self.agent.chat(
                    "Fill missing values: "
                    "- Numeric columns: median "
                    "- Categorical columns: mode "
                    "- Datetime columns: most frequent "
                    "and return the modified dataframe"
                    """
                    - DO NOT use pd.api or any advanced pandas methods
                    - DO NOT use try/except blocks or complex error handling
                    - DO NOT use inplace=True parameter anywhere
                    - DO NOT use df.astype() with parameters that might be restricted
                    - Use only the most basic pandas operations
                    - Preserve all missing values exactly as they were
                    - Use direct assignment for all changes: df['COL'] = df['COL'].operation()"""
                )
                
            elif strategy == 'drop':
                final_df = self.agent.chat("Drop all rows with any missing values")
            
            else:  # Custom strategy
                final_df = self.agent.chat(f"Handle missing values by {strategy}")
            
            # Verify changes
            if not verbose:
                print("\nMissing values before:")
                print(original.isna().sum())
            
            # Get the current state after modifications
            # Use the final result from chat
            if isinstance(final_df, pd.DataFrame):
                current_df = final_df
            else:
                # If the result isn't a dataframe, try to get the current dataframe from agent
                current_df = self.agent.dataframe.execute()
            
            if not verbose:
                    print("\nMissing values after:")
                    print(current_df.isna().sum())
            if current_df.isna().sum().sum() == 0:
                if not verbose:
                    print("No missing values left in the dataframe.")
                
            else:
                if not verbose:
                    print("There are still missing values in the dataframe.")

        if self.agent.dataframe.execute().isna().sum().sum() != 0:
            """
            If the missing values are still present after the specified number of tries,
            Hard code method to initiate it
            """
            pass
            
        
    def save_dataframe_to_csv(self, original_dataset_id, filepath=None, index=False):
        """
        Save the current dataframe to a CSV file and store it in the database.
        
        Args:
            original_dataset_id (int): ID of the original dataset
            filepath (str, optional): Path to save the CSV file. If None, a temporary path will be used.
            index (bool): Whether to include the index in the CSV
        
        Returns:
            Dataset: The newly created Dataset object
        """
        try:
            original_dataset = Dataset.objects.get(id=original_dataset_id)
            
            # Get the current dataframe from the agent
            current_df = self.agent.dataframe.execute()
            
            # Create a temp file path if not provided
            if filepath is None:
                import tempfile
                import os
                temp_dir = tempfile.gettempdir()
                filename = f"cleaned_{original_dataset.name.split('.')[0]}_{int(time.time())}.csv"
                filepath = os.path.join(temp_dir, filename)
            
            # Save to CSV
            current_df.to_csv(filepath, index=index)
            
            # Create a Django File object from the saved CSV
            from django.core.files import File
            with open(filepath, 'rb') as file_obj:
                # Create new dataset with reference to the processed file
                cleaned_name = f"{os.path.splitext(original_dataset.name)[0]}_cleaned"
                new_dataset = Dataset.objects.create(
                    name=cleaned_name,
                    file=File(file_obj, name=os.path.basename(filepath))
                )
            
            # Optionally remove the temp file if it was created temporarily
            if filepath.startswith(tempfile.gettempdir()):
                os.remove(filepath)
                
            return new_dataset  # Return the Dataset object, not a Response
            
        except Exception as e:
            print(f"Error saving processed dataframe: {str(e)}")
            raise e  # Re-raise to be handled by caller

    def get_dataframe(self):
        """
        Get the current state of the dataframe.
        """
        return self.agent.dataframe.execute()
    def sample_data(self):
        """
        Get a sample of the dataframe and format it for frontend consumption.
        """
        # Get 5 sample rows
        sample_df = self.agent.dataframe.execute().sample(5).reset_index()
        
        # Convert to list of dictionaries format expected by frontend
        formatted_samples = []
        for i, row in sample_df.iterrows():
            # Create a dictionary for each row
            row_dict = {'id': i+1}  # Start id from 1
            
            # Add all columns from the dataframe
            for column in sample_df.columns:
                if column != 'index':  # Skip the index column
                    # Convert numpy/pandas types to Python native types for JSON serialization
                    value = row[column]
                    if hasattr(value, 'item'):  # Check if it's a numpy type
                        value = value.item()
                    row_dict[column] = value
                    
            formatted_samples.append(row_dict)
        
        return formatted_samples