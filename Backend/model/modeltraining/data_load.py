from .library import *

class DataLoader:
    def __init__(self,data,target_column=None):
        self.file_path = data
        self.df = None
        self.target_column = target_column
        self.problem_type = None
        self.label_encoder = None
        self.preprocessor = None
        self.feature_names = None
        self.x=None
        self.y=None

    def load_data(self):
        """Load data from file with robust error handling."""
        print(f"\nLoading data from: {self.file_path}")
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(self.file_path)
            else:
                self.df = pd.read_csv(self.file_path)

            self.df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '') for col in self.df.columns]
            print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print("\nData preview:")
            print(self.df.head(3))
            print("\nData info:")
            print(self.df.info())
            return True

        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            traceback.print_exc()
            return False

    def _check_data_issues(self, df):
        missing_vals = df.isnull().sum()
        if missing_vals.sum() > 0:
            print("\nWarning: Missing values in columns:")
            print(missing_vals[missing_vals > 0])

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\nWarning: {duplicates} duplicate rows")

        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            print(f"\nWarning: Constant columns detected: {constant_cols}")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values > 50:
                print(f"Warning: High cardinality in '{col}' ({unique_values} unique values)")

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr = [(numeric_cols[i], numeric_cols[j], upper.iloc[i, j])
                             for i in range(len(numeric_cols)) for j in range(i + 1, len(numeric_cols))
                             if upper.iloc[i, j] > 0.9]
                if high_corr:
                    print("\nWarning: Highly correlated numeric features:")
                    for col1, col2, val in high_corr:
                        print(f"  - {col1} and {col2}: {val:.2f}")
            except Exception as e:
                print(f"\nCorrelation check error: {str(e)}")

    def set_target(self):
        unique_values = self.df[self.target_column].nunique()

        if self.df[self.target_column].dtype == 'object' or (unique_values < 10 and unique_values < 0.1 * len(self.df)):
            self.problem_type = 'classification'
            print(f"\nClassification problem with {unique_values} classes")
            self.label_encoder = LabelEncoder()
            self.df[self.target_column] = self.label_encoder.fit_transform(self.df[self.target_column])
            print(f"Target encoded: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            class_counts = self.df[self.target_column].value_counts()
            if len(class_counts) > 1 and (class_counts.min() < 5 or class_counts.min() / class_counts.max() < 0.1):
                print("\nWarning: Class imbalance")
                print(class_counts)
        else:
            self.problem_type = 'regression'
            self.label_encoder = None
            print(f"\nRegression problem for '{self.target_column}'")

        print("\nTarget distribution:")
        print(self.df[self.target_column].describe())

    def prepare_data(self):
        print("\nPreprocessing data...")
        if self.df[self.target_column].isna().sum() > 0:
            print("Removing rows with missing target values")
            self.df = self.df.dropna(subset=[self.target_column])

        self.x = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

        numeric_transformer = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('transform', PowerTransformer(method='yeo-johnson'))
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
        ], remainder='passthrough')

        print("Fitting preprocessing pipeline...")
        X_transformed = self.preprocessor.fit_transform(self.x)
        print(f"Final feature count: {X_transformed.shape[1]}")

        try:
            num_features = self.preprocessor.transformers_[0][2]
            cat_features = self.preprocessor.transformers_[1][2]
            cat_encoder = self.preprocessor.transformers_[1][1].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(cat_features) if hasattr(cat_encoder, 'get_feature_names_out') else []
            self.feature_names = np.concatenate([num_features, cat_feature_names])
            print(f"\nExtracted {len(self.feature_names)} feature names")
        except Exception as e:
            print(f"\nCould not extract feature names: {str(e)}")
            self.feature_names = None

        return self.x, self.y