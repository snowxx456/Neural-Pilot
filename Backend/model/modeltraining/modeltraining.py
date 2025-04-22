import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import traceback
import time
from functools import partial
from scipy import stats
import joblib
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from .targetcolumn import file_path, target_column

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class AdvancedMLPipeline:
    def __init__(self):
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.problem_type = None
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.X_test = None
        self.y_test = None
        self.df = None
        self.target_column = None
        
    def load_data(self, file_path):
        """Load data from CSV file and return a pandas DataFrame with robust error handling."""
        print(f"\nLoading data from: {file_path}")
        try:
            # Try to detect the file extension and load accordingly
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                # Default to CSV if extension is unclear
                df = pd.read_csv(file_path)
            
            # Clean column names (remove spaces, special chars, etc.)
            df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '') 
                          for col in df.columns]
            
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            print("\nData preview:")
            print(df.head(3))
            print("\nData info:")
            print(df.info())
            
            # Store the dataframe
            self.df = df
            
            # Check for issues with data that might cause problems later
            self._check_data_issues(df)
            
            return True
            
        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            traceback.print_exc()
            return False
    
    def _check_data_issues(self, df):
        """Check for common data issues and warn user."""
        # Check for missing values
        missing_vals = df.isnull().sum()
        if missing_vals.sum() > 0:
            print("\nWarning: Missing values detected in the following columns:")
            print(missing_vals[missing_vals > 0])
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\nWarning: {duplicates} duplicate rows detected in the dataset.")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            print(f"\nWarning: Constant columns detected (no variance): {constant_cols}")
        
        # Check for high cardinality categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values > 50:  # Threshold can be adjusted
                print(f"Warning: Column '{col}' has high cardinality ({unique_values} unique values)")
        
        # Check for highly correlated numerical features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:  # Only check if there's more than one numeric column
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr = [(col1, col2, corr_val) for col1, col2, corr_val in 
                             zip(*np.where(upper > 0.9), upper[upper > 0.9])]
                if high_corr:
                    print("\nWarning: High correlation (>0.9) detected between features:")
                    for col1, col2, corr_val in high_corr:
                        print(f"  - {numeric_cols[col1]} and {numeric_cols[col2]}: {corr_val:.2f}")
            except Exception as e:
                print(f"\nCould not check feature correlations: {str(e)}")
    
    def set_target(self, target_column):
        """Set the target column for modeling."""
        if target_column not in self.df.columns:
            print(f"Error: '{target_column}' is not a valid column in the dataset.")
            return False
        
        self.target_column = target_column
        
        # Determine problem type
        unique_values = self.df[target_column].nunique()
        if self.df[target_column].dtype == 'object' or (unique_values < 10 and unique_values < 0.1 * len(self.df)):
            self.problem_type = 'classification'
            print(f"\nClassification problem detected with {unique_values} classes")
            
            # Create label encoder for target column
            self.label_encoder = LabelEncoder()
            self.df[target_column] = self.label_encoder.fit_transform(self.df[target_column])
            class_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
            print(f"Target column encoded: {class_mapping}")
            
            # Check for class imbalance
            class_counts = self.df[target_column].value_counts()
            if len(class_counts) > 1 and (class_counts.min() < 5 or class_counts.min()/class_counts.max() < 0.1):
                print("\nWarning: Significant class imbalance detected")
                print(class_counts)
        else:
            self.problem_type = 'regression'
            self.label_encoder = None
            print(f"\nRegression problem detected for continuous target '{target_column}'")
        
        print("\nTarget distribution:")
        print(self.df[target_column].describe())
        
        return True
    
    def prepare_data(self):
        """Prepare data for ML by handling missing values and encoding categorical features."""
        print("\nStarting data preprocessing...")
        
        # Handle missing values in the target variable
        if self.df[self.target_column].isna().sum() > 0:
            print(f"Warning: {self.df[self.target_column].isna().sum()} missing values in target column. Removing these rows.")
            self.df = self.df.dropna(subset=[self.target_column])
        
        # Split features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Create preprocessing pipelines for both numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),  # More sophisticated imputation
            ('scaler', StandardScaler()),
            ('transform', PowerTransformer(method='yeo-johnson'))  # Handle skewness
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
                ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
            ],
            remainder='passthrough'  # Include columns not specified in transformers
        )
        
        print("\nFitting preprocessing pipeline...")
        
        # Apply preprocessing to get final feature count
        X_transformed = self.preprocessor.fit_transform(X)
        print(f"Final feature count: {X_transformed.shape[1]}")
        
        # Try to get feature names if possible
        try:
            # Feature names for numerical columns
            num_features = self.preprocessor.transformers_[0][2]
            
            # Feature names for categorical columns (after one-hot encoding)
            cat_features = self.preprocessor.transformers_[1][2]
            cat_encoder = self.preprocessor.transformers_[1][1].named_steps['onehot']
            if hasattr(cat_encoder, 'get_feature_names_out') and len(cat_features) > 0:
                cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
            else:
                cat_feature_names = []
            
            # Combine feature names
            self.feature_names = np.concatenate([num_features, cat_feature_names])
            print(f"\nFeature names successfully extracted: {len(self.feature_names)} features")
        except Exception as e:
            print(f"\nCould not extract feature names: {str(e)}")
            self.feature_names = None
        
        return X, y
    
    def _get_models(self):
        """Return appropriate models based on problem type."""
        if self.problem_type == 'classification':
            models = {
                'LogisticRegression': LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced'),
                'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
                'LightGBM': LGBMClassifier(random_state=42, class_weight='balanced'),
                'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
                'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
                'KNN': KNeighborsClassifier(),
                'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                'MLP': MLPClassifier(random_state=42, max_iter=1000),
                'NaiveBayes': GaussianNB()
            }
        else:  # regression
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'ElasticNet': ElasticNet(random_state=42),
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'LightGBM': LGBMRegressor(random_state=42),
                'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'DecisionTree': DecisionTreeRegressor(random_state=42),
                'MLP': MLPRegressor(random_state=42, max_iter=1000),
                'ExtraTrees': ExtraTreesRegressor(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42),
                'SGDRegressor': SGDRegressor(random_state=42)
            }
        return models
    
    def _get_hyperparameters(self, model_name):
        """Return hyperparameter grids for different models."""
        if self.problem_type == 'classification':
            params = {
                'LogisticRegression': {
                    'model__C': Real(1e-3, 1e3, prior='log-uniform'),
                    'model__solver': Categorical(['liblinear', 'saga']),
                    'model__penalty': Categorical(['l1', 'l2']),
                    'model__class_weight': Categorical([None, 'balanced'])
                },
                'RandomForest': {
                    'model__n_estimators': Integer(50, 500),
                    'model__max_depth': Integer(3, 30),
                    'model__min_samples_split': Integer(2, 20),
                    'model__min_samples_leaf': Integer(1, 10),
                    'model__class_weight': Categorical([None, 'balanced'])
                },
                'GradientBoosting': {
                    'model__n_estimators': Integer(50, 500),
                    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'model__max_depth': Integer(3, 10),
                    'model__min_samples_split': Integer(2, 20),
                    'model__min_samples_leaf': Integer(1, 10)
                },
                'XGBoost': {
                    'model__n_estimators': Integer(50, 500),
                    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'model__max_depth': Integer(3, 10),
                    'model__subsample': Real(0.5, 1.0),
                    'model__colsample_bytree': Real(0.5, 1.0),
                    'model__gamma': Real(0, 5),
                    'model__reg_alpha': Real(0, 10),
                    'model__reg_lambda': Real(0, 10)
                },
                'LightGBM': {
                    'model__n_estimators': Integer(50, 500),
                    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'model__num_leaves': Integer(20, 100),
                    'model__max_depth': Integer(3, 15),
                    'model__min_child_samples': Integer(5, 50),
                    'model__subsample': Real(0.5, 1.0),
                    'model__colsample_bytree': Real(0.5, 1.0),
                    'model__reg_alpha': Real(0, 10),
                    'model__reg_lambda': Real(0, 10)
                },
                'SVM': {
                    'model__C': Real(1e-3, 1e3, prior='log-uniform'),
                    'model__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                    'model__gamma': Categorical(['scale', 'auto']),
                    'model__class_weight': Categorical([None, 'balanced'])
                },
                'MLP': {
                    'model__hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)]),
                    'model__activation': Categorical(['relu', 'tanh', 'logistic']),
                    'model__alpha': Real(1e-5, 1e-1, prior='log-uniform'),
                    'model__learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform')
                }
            }
        else:  # regression
            params = {
                'LinearRegression': {},
                'Ridge': {
                    'model__alpha': Real(1e-3, 1e3, prior='log-uniform')
                },
                'Lasso': {
                    'model__alpha': Real(1e-3, 1e3, prior='log-uniform')
                },
                'ElasticNet': {
                    'model__alpha': Real(1e-3, 1e3, prior='log-uniform'),
                    'model__l1_ratio': Real(0, 1)
                },
                'RandomForest': {
                    'model__n_estimators': Integer(50, 500),
                    'model__max_depth': Integer(3, 30),
                    'model__min_samples_split': Integer(2, 20),
                    'model__min_samples_leaf': Integer(1, 10)
                },
                'GradientBoosting': {
                    'model__n_estimators': Integer(50, 500),
                    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'model__max_depth': Integer(3, 10),
                    'model__min_samples_split': Integer(2, 20),
                    'model__min_samples_leaf': Integer(1, 10)
                },
                'XGBoost': {
                    'model__n_estimators': Integer(50, 500),
                    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'model__max_depth': Integer(3, 10),
                    'model__subsample': Real(0.5, 1.0),
                    'model__colsample_bytree': Real(0.5, 1.0),
                    'model__gamma': Real(0, 5),
                    'model__reg_alpha': Real(0, 10),
                    'model__reg_lambda': Real(0, 10)
                },
                'LightGBM': {
                    'model__n_estimators': Integer(50, 500),
                    'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'model__num_leaves': Integer(20, 100),
                    'model__max_depth': Integer(3, 15),
                    'model__min_child_samples': Integer(5, 50),
                    'model__subsample': Real(0.5, 1.0),
                    'model__colsample_bytree': Real(0.5, 1.0),
                    'model__reg_alpha': Real(0, 10),
                    'model__reg_lambda': Real(0, 10)
                },
                'SVR': {
                    'model__C': Real(1e-3, 1e3, prior='log-uniform'),
                    'model__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                    'model__gamma': Categorical(['scale', 'auto']),
                    'model__epsilon': Real(0.01, 1.0)
                },
                'MLP': {
                    'model__hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)]),
                    'model__activation': Categorical(['relu', 'tanh', 'logistic']),
                    'model__alpha': Real(1e-5, 1e-1, prior='log-uniform'),
                    'model__learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform')
                }
            }
        
        # Return the parameters for the specific model or empty dict if not found
        return params.get(model_name, {})
    
    def train_models(self, X, y):
        """Train and evaluate multiple ML models with robust error handling."""
        # Split the data
        if self.problem_type == 'classification':
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                print(f"Could not stratify split due to class imbalance. Using random split instead: {str(e)}")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\nTraining set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        
        # Get models to try based on problem type
        models = self._get_models()
        
        # Train each model
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            
            try:
                # Create pipeline with preprocessor and model
                pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('model', model)
                ])
                
                # Cross-validation scoring
                scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
                
                # Perform cross-validation
                start_time = time.time()
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
                cv_time = time.time() - start_time
                
                if self.problem_type == 'classification':
                    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (time: {cv_time:.2f}s)")
                else:
                    print(f"Cross-validation R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (time: {cv_time:.2f}s)")
                
                # Train the model on the full training set
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Evaluate the model
                if self.problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred)
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    
                    # For probability-based metrics (ROC, PR curve)
                    y_proba = None
                    if hasattr(pipeline, "predict_proba"):
                        try:
                            y_proba = pipeline.predict_proba(X_test)[:, 1]
                        except:
                            pass  # Some models might not support predict_proba
                else:  # regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    evs = explained_variance_score(y_test, y_pred)
                    
                    report = (f"Mean Squared Error: {mse:.4f}\n"
                             f"Root Mean Squared Error: {rmse:.4f}\n"
                             f"Mean Absolute Error: {mae:.4f}\n"
                             f"R2 Score: {r2:.4f}\n"
                             f"Explained Variance Score: {evs:.4f}")
                    conf_matrix = None
                    y_proba = None
                    accuracy = r2  # Use R2 as the primary metric for regression
                
                # Store results
                self.results[name] = {
                    'pipeline': pipeline,
                    'accuracy': accuracy,
                    'cv_scores': cv_scores,
                    'report': report,
                    'conf_matrix': conf_matrix,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'true_values': y_test,
                    'train_time': train_time,
                    'cv_time': cv_time
                }
                
                # Print results
                if self.problem_type == 'classification':
                    print(f"Test accuracy: {accuracy:.4f} (train time: {train_time:.2f}s)")
                else:
                    print(f"Test R2: {accuracy:.4f} (train time: {train_time:.2f}s)")
                print("\nPerformance Report:")
                print(report)
                
            except Exception as e:
                print(f"\nError training {name}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Determine the best model
        if self.results:
            self.best_model_name = max(self.results, key=lambda k: self.results[k]['accuracy'])
            self.best_model = self.results[self.best_model_name]
            print(f"\nBest model: {self.best_model_name} with {'accuracy' if self.problem_type == 'classification' else 'R2'}: {self.best_model['accuracy']:.4f}")
        
        return self.results
    
    def hypertune_best_model(self, X, y):
        """Hypertune the best model with Bayesian optimization."""
        if not self.best_model_name:
            print("No best model identified for hyperparameter tuning.")
            return None
        
        print(f"\n{'='*50}")
        print(f"Hyperparameter tuning for {self.best_model_name}...")
        
        # Get the model and its hyperparameter grid
        models = self._get_models()
        model = models[self.best_model_name]
        param_grid = self._get_hyperparameters(self.best_model_name)
        
        if not param_grid:
            print(f"No hyperparameter grid defined for {self.best_model_name}")
            return None
        
        # Create pipeline with the best model
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])
        
        # Split data for hyperparameter tuning
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Set up scoring
        scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
        
        # Use Bayesian optimization for hyperparameter tuning
        opt = BayesSearchCV(
            pipeline,
            param_grid,
            n_iter=50,  # Number of iterations
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        try:
            # Fit the optimizer
            start_time = time.time()
            opt.fit(X_train, y_train)
            tune_time = time.time() - start_time
            
            # Get best parameters and model
            best_params = opt.best_params_
            best_score = opt.best_score_
            
            print(f"\nBest cross-validation score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
            print(f"Tuning time: {tune_time:.2f} seconds")
            
            # Evaluate on validation set
            y_pred = opt.predict(X_val)
            
            if self.problem_type == 'classification':
                accuracy = accuracy_score(y_val, y_pred)
                report = classification_report(y_val, y_pred)
                print(f"\nValidation accuracy with tuned model: {accuracy:.4f}")
            else:
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                evs = explained_variance_score(y_val, y_pred)
                
                report = (f"\nMean Squared Error: {mse:.4f}\n"
                         f"Root Mean Squared Error: {rmse:.4f}\n"
                         f"Mean Absolute Error: {mae:.4f}\n"
                         f"R2 Score: {r2:.4f}\n"
                         f"Explained Variance Score: {evs:.4f}")
                print(f"\nValidation R2 with tuned model: {r2:.4f}")
            
            print("\nPerformance report:")
            print(report)
            
            # Update the best model with the tuned version
            self.best_model['pipeline'] = opt.best_estimator_
            self.best_model['accuracy'] = accuracy if self.problem_type == 'classification' else r2
            self.best_model['best_params'] = best_params
            
            return opt.best_estimator_, best_params, best_score
            
        except Exception as e:
            print(f"\nError during hyperparameter tuning: {str(e)}")
            traceback.print_exc()
            return None, None, None
    
    def visualize_results(self):
        """Visualize model results and feature importance."""
        if not self.results:
            print("No models were successfully trained. Cannot visualize results.")
            return
        
        # Create visualizations directory if it doesn't exist
        if not os.path.exists('ml_visualizations'):
            os.makedirs('ml_visualizations')
        
        # Get class names for visualization labels (classification only)
        if self.problem_type == 'classification' and self.label_encoder:
            class_names = self.label_encoder.classes_
        else:
            class_names = None
        
        # Plot confusion matrix for classification
        if self.problem_type == 'classification' and self.best_model['conf_matrix'] is not None:
            plt.figure(figsize=(10, 8))
            cm = self.best_model['conf_matrix']
            
            # Calculate percentages for annotation
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100
            annot = np.empty_like(cm).astype(str)
            nrows, ncols = cm.shape
            for i in range(nrows):
                for j in range(ncols):
                    annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"
            
            sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names,
                        cbar=False, linewidths=0.5, linecolor='black')
            plt.title(f'Confusion Matrix - {self.best_model_name}\n', fontsize=14)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('ml_visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot ROC curve if probabilities are available
            if self.best_model['probabilities'] is not None:
                plt.figure(figsize=(10, 8))
                fpr, tpr, _ = roc_curve(self.y_test, self.best_model['probabilities'])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title(f'ROC Curve - {self.best_model_name}\n', fontsize=14)
                plt.legend(loc="lower right", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.savefig('ml_visualizations/roc_curve.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot Precision-Recall curve
                plt.figure(figsize=(10, 8))
                precision, recall, _ = precision_recall_curve(self.y_test, self.best_model['probabilities'])
                avg_precision = np.mean(precision)
                
                plt.plot(recall, precision, color='blue', lw=2, 
                         label=f'Avg Precision = {avg_precision:.2f}')
                plt.xlabel('Recall', fontsize=12)
                plt.ylabel('Precision', fontsize=12)
                plt.title(f'Precision-Recall Curve - {self.best_model_name}\n', fontsize=14)
                plt.legend(loc="upper right", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.savefig('ml_visualizations/precision_recall_curve.png', dpi=300, bbox_inches='tight')
                plt.close()
        elif self.problem_type == 'regression':
            # Plot actual vs predicted for regression
            plt.figure(figsize=(10, 8))
            plt.scatter(self.y_test, self.best_model['predictions'], alpha=0.6, edgecolors='w', s=80)
            plt.plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'k--', lw=2)
            plt.xlabel('Actual Values', fontsize=12)
            plt.ylabel('Predicted Values', fontsize=12)
            plt.title(f'Actual vs Predicted - {self.best_model_name}\n', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('ml_visualizations/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot residuals
            plt.figure(figsize=(10, 8))
            residuals = self.y_test - self.best_model['predictions']
            plt.scatter(self.best_model['predictions'], residuals, alpha=0.6, edgecolors='w', s=80)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.xlabel('Predicted Values', fontsize=12)
            plt.ylabel('Residuals', fontsize=12)
            plt.title(f'Residual Plot - {self.best_model_name}\n', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('ml_visualizations/residual_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot feature importance for tree-based models
        if self.best_model_name in ['RandomForest', 'GradientBoosting', 'RandomForestRegressor', 
                                   'GradientBoostingRegressor', 'XGBoost', 'LightGBM', 'CatBoost']:
            try:
                # Get feature importances
                model = self.best_model['pipeline'].named_steps['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    raise AttributeError("Model doesn't have feature_importances_ or coef_ attribute")
                
                # Get feature names if available
                if self.feature_names is not None and len(self.feature_names) == len(importances):
                    feature_names = self.feature_names
                else:
                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                
                # Sort feature importances
                sorted_indices = np.argsort(importances)[::-1]
                sorted_importances = importances[sorted_indices]
                sorted_features = np.array(feature_names)[sorted_indices]
                
                # Plot top N features
                top_n = min(20, len(feature_names))  # Show at most 20 features
                
                plt.figure(figsize=(12, 8))
                plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}\n', fontsize=14)
                bars = plt.barh(range(top_n), sorted_importances[:top_n], align='center', color='skyblue')
                plt.yticks(range(top_n), sorted_features[:top_n], fontsize=10)
                plt.gca().invert_yaxis()  # Highest importance at the top
                plt.xlabel('Importance Score', fontsize=12)
                plt.grid(True, alpha=0.3, axis='x')
                
                # Add value labels to bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.01 * max(sorted_importances[:top_n]), 
                             bar.get_y() + bar.get_height()/2,
                             f'{width:.3f}', 
                             va='center', ha='left', fontsize=9)
                
                plt.tight_layout()
                plt.savefig('ml_visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Print top features and their importances
                print("\nTop 10 important features:")
                for i in range(min(10, len(feature_names))):
                    print(f"{sorted_features[i]}: {sorted_importances[i]:.4f}")
            except Exception as e:
                print(f"\nCould not plot feature importance: {str(e)}")
        
        # Plot model comparison
        plt.figure(figsize=(14, 8))
        model_names = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in model_names]
        cv_means = [self.results[model]['cv_scores'].mean() for model in model_names]
        train_times = [self.results[model]['train_time'] for model in model_names]
        
        # Create grouped bar chart
        x = np.arange(len(model_names))
        width = 0.35
        
        metric_name = 'Accuracy' if self.problem_type == 'classification' else 'R2 Score'
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot accuracy/R2 scores
        bars1 = ax1.bar(x - width/2, accuracies, width, label=f'Test {metric_name}', color='skyblue')
        bars2 = ax1.bar(x + width/2, cv_means, width, label=f'CV {metric_name}', color='lightgreen')
        
        # Add training times as line plot
        ax2 = ax1.twinx()
        line = ax2.plot(x, train_times, color='red', marker='o', label='Training Time (s)')
        
        ax1.set_title(f'Model Performance Comparison\n', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax1.set_ylabel(metric_name, fontsize=12)
        ax1.set_ylim(0, 1.1 if self.problem_type == 'classification' else None)
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.set_ylabel('Training Time (seconds)', fontsize=12)
        ax2.set_ylim(0, max(train_times) * 1.1)
        
        # Combine legends
        lines = [bars1, bars2, line[0]]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=3, fontsize=10)
        
        # Add accuracy labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('ml_visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        if self.problem_type == 'classification' and self.label_encoder:
            # For categorical targets, map back to original labels
            target_counts = self.df[self.target_column].value_counts()
            if len(target_counts) <= 10:  # Only plot if reasonable number of classes
                bars = plt.bar(range(len(target_counts)), target_counts.values, color='skyblue')
                plt.xticks(range(len(target_counts)), target_counts.index, rotation=45, ha='right')
                plt.title(f'Distribution of Target Variable\n', fontsize=14)
                plt.ylabel('Count', fontsize=12)
                
                # Add counts on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(target_counts.values),
                             str(int(height)), ha='center', va='bottom')
        else:
            # For numerical targets
            sns.histplot(self.df[self.target_column], kde=True, color='skyblue')
            plt.title(f'Distribution of Target Variable\n', fontsize=14)
            plt.ylabel('Count', fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ml_visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot correlation heatmap for numerical features
        try:
            numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
            if len(numeric_df.columns) > 1:  # Only if we have at least 2 numeric columns
                plt.figure(figsize=(12, 10))
                corr = numeric_df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                
                sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", 
                           linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
                plt.title('Feature Correlation Heatmap\n', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig('ml_visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"\nCould not create correlation heatmap: {str(e)}")
        
        print("\nVisualization complete. Results saved in 'ml_visualizations' folder.")
    
    def save_model(self, filename='best_model.pkl'):
        """Save the best model to a file."""
        if self.best_model is None:
            print("No best model to save.")
            return False
        
        try:
            # Save the entire pipeline (including preprocessor)
            joblib.dump(self.best_model['pipeline'], filename)
            print(f"\nBest model saved to {filename}")
            return True
        except Exception as e:
            print(f"\nError saving model: {str(e)}")
            return False
    
    def run_pipeline(self, file_path, target_column):
        """Run the complete ML pipeline with specified parameters."""
        # Step 1: Load data
        if not self.load_data(file_path):
            return False
        
        # Step 2: Set target column
        if not self.set_target(target_column):
            return False
        
        # Rest of your existing pipeline implementation...
        X, y = self.prepare_data()
        self.train_models(X, y)
        self.visualize_results()
        
        # Optional tuning
        tune_choice = input("\nPerform hyperparameter tuning? (y/n): ").lower()
        if tune_choice == 'y':
            self.hypertune_best_model(X, y)
            self.visualize_results()
        
        # Optional model saving
        save_choice = input("\nSave the best model? (y/n): ").lower()
        if save_choice == 'y':
            model_name = input("Enter filename (default: best_model.pkl): ") or "best_model.pkl"
            self.save_model(model_name)
        
        print("\nPipeline execution complete!")
        return True

def main():
    """Main function to run the ML pipeline."""
    try:
        print("="*70)
        print("Advanced Machine Learning Pipeline")
        print("="*70)
        
        if not file_path:
            print("Error: No file path specified in t.py")
            return
            
        if not target_column:
            print("Error: Could not determine target column automatically")
            print("Please specify target column manually in t.py")
            return
        
        # Create pipeline instance
        pipeline = AdvancedMLPipeline()
        
        print("\nStarting pipeline with:")
        print(f"Data file: {file_path}")
        print(f"Target column: {target_column}")
        
        # Run the pipeline
        success = pipeline.run_pipeline(file_path, target_column)
        
        if not success:
            print("Pipeline execution failed")
    
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()