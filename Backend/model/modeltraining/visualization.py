from .library import *

class VisualizationHandler():
    def __init__(self,data, best_model, best_model_name, results,feature_names,label_encoder,
                probabilities=None, y_test=None, problem_type=None):
        self.df = data
        self.best_model_name = best_model_name
        self.best_model = best_model
        self.results = results
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.probabilities = probabilities
        self.y_test = y_test
        self.problem_type = problem_type

    def correlation_graph(self):
            try:
                numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
                if len(numeric_df.columns) <= 1:
                    return {"error": "Not enough numeric columns to calculate correlation"}
                
                # Calculate correlation matrix
                corr = numeric_df.corr()
                
                # Create the correlation data structure
                correlation_data = {
                    "features": list(corr.columns),
                    "matrix": []
                }
                
                # Convert correlation matrix to list of lists (2D array)
                for _, row in corr.iterrows():
                    # Round to 2 decimal places
                    correlation_data["matrix"].append([round(float(val), 2) for val in row.values])
                
                return correlation_data
                
            except Exception as e:
                return {"error": f"Could not create correlation data: {str(e)}"}
            
    def feature_importance(self):
        if self.best_model_name in ['RandomForest', 'GradientBoosting', 'RandomForestRegressor', 
                                'GradientBoostingRegressor', 'XGBoost', 'LightGBM', 'CatBoost']:
            try:
                # Get feature importances
                model = self.best_model['pipeline'].named_steps['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                    # Handle multi-class case where coef_ is 2D
                    if importances.ndim > 1:
                        importances = np.mean(importances, axis=0)
                else:
                    return {"error": "Model doesn't have feature_importances_ or coef_ attribute"}
                
                # Get feature names if available
                if self.feature_names is not None and len(self.feature_names) == len(importances):
                    features = self.feature_names
                else:
                    features = [f"Feature {i}" for i in range(len(importances))]
                
                # Sort feature importances
                sorted_indices = np.argsort(importances)[::-1]
                sorted_importances = importances[sorted_indices]
                sorted_features = np.array(features)[sorted_indices]
                
                # Format data for frontend
                top_n = min(20, len(features))  # Show at most 20 features
                feature_importance = []
                
                for i in range(top_n):
                    feature_importance.append({
                        "feature": sorted_features[i],
                        "importance": round(float(sorted_importances[i]), 3)
                    })
                
                return feature_importance
                
            except Exception as e:
                return {"error": f"Could not extract feature importance: {str(e)}"}
    
    def confusion_matrix(self):
        try:
            # Check if confusion matrix exists in the best model
            if self.best_model is None or 'conf_matrix' not in self.best_model or self.best_model['conf_matrix'] is None:
                return {"error": "No confusion matrix available"}
            
            # Get the confusion matrix from best model
            cm = np.array(self.best_model['conf_matrix'])
            
            # Calculate percentages
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = np.round(cm / cm_sum.astype(float) * 100, 1).tolist()
            
            # Set class names based on label encoder
            if self.problem_type == 'classification' and hasattr(self, 'label_encoder') and self.label_encoder:
                class_names = list(self.label_encoder.classes_)
            else:
                class_names = [f"Class {i}" for i in range(cm.shape[0])]
            
            # Format the data for frontend
            confusion_matrix_data = {
                "matrix": cm.tolist(),
                "percentages": cm_perc,
                "classNames": class_names,
                "modelName": self.best_model_name,
            }
            
            return confusion_matrix_data
        
        except Exception as e:
            return {"error": f"Could not extract confusion matrix data: {str(e)}"}
        
    def roc_curve(self):
    # First, check if we have the necessary data
        if self.probabilities is None or self.y_test is None:
            return {"error": "Missing probabilities or test data for ROC curve"}
            
        if self.best_model_name in ['RandomForest', 'GradientBoosting', 'RandomForestRegressor', 
                                'GradientBoostingRegressor', 'XGBoost', 'LightGBM', 'CatBoost']:
            try:
                # Calculate ROC curve and ROC area
                fpr, tpr, _ = roc_curve(self.y_test, self.probabilities)
                roc_auc = auc(fpr, tpr)
                
                # Format data for frontend
                # Convert numpy arrays to lists and round to 4 decimal places
                roc_data = {
                    "fpr": [round(float(x), 4) for x in fpr],
                    "tpr": [round(float(x), 4) for x in tpr],
                    "auc": round(float(roc_auc), 4),
                    "modelName": self.best_model_name
                }
                
                return roc_data
                    
            except Exception as e:
                return {"error": f"Could not extract ROC curve data: {str(e)}"}
        return {"error": "Current model type doesn't support ROC curve visualization"}
    
    def precision_recall_curve(self):
        if self.best_model_name in ['RandomForest', 'GradientBoosting', 'RandomForestRegressor', 
                                'GradientBoostingRegressor', 'XGBoost', 'LightGBM', 'CatBoost']:
            try:
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(self.y_test, self.probabilities)
                avg_precision = np.mean(precision)
                
                # Format data for frontend
                # Convert numpy arrays to lists and round to 4 decimal places
                pr_data = {
                    "precision": [round(float(x), 4) for x in precision],
                    "recall": [round(float(x), 4) for x in recall],
                    "avgPrecision": round(float(avg_precision), 4),
                    "modelName": self.best_model
                }
                
                return pr_data
                
            except Exception as e:
                return {"error": f"Could not extract Precision-Recall curve data: {str(e)}"}

    def compare_model_plot(self):  
        try:
            model_names = list(self.results.keys())
            
            # Metrics based on problem type
            metric_name = 'Accuracy' if self.problem_type == 'classification' else 'R2 Score'
            
            # Prepare data for frontend
            comparison_data = {
                "model_names": model_names,
                "Accuracy": [self.results[model]['accuracy'] for model in model_names],
                "Precision": [self.results[model]['precision'] for model in model_names],
                "Recall": [self.results[model]['recall'] for model in model_names],
                "F1 Score": [self.results[model]['f1_score'] for model in model_names],
                "train_times": [self.results[model]['train_time'] for model in model_names],
                "metric_name": metric_name,
                "problem_type": self.problem_type
            }
            
            return comparison_data
            
        except Exception as e:
            return {"error": f"Could not prepare model comparison data: {str(e)}"}
    
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