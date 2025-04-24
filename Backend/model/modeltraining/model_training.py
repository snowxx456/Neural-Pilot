from library import *


class ModelTrainer ():
    def __init__(self, problem_type,target_column,data,x,y,preprocessor=None):
        self.problem_type = problem_type
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.X_test = None
        self.y_test = None
        self.x = x
        self.y = y
        self.df = data
        self.target_column = target_column
        self.preprocessor = preprocessor

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
    
    def train_models(self):
        """Train and evaluate multiple ML models with robust error handling."""
        # Split the data
        if self.problem_type == 'classification':
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.x, self.y, test_size=0.2, random_state=42, stratify=self.y
                )
            except ValueError as e:
                print(f"Could not stratify split due to class imbalance. Using random split instead: {str(e)}")
                X_train, X_test, y_train, y_test = train_test_split(
                    self.x, self.y, test_size=0.2, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=0.2, random_state=42
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
    
    def hypertune_best_model(self):
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
        X_train, X_val, y_train, y_val = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        
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
    
    def get_formatted_results(self):
        """Convert training results to match the frontend's ModelResult[] type"""
        formatted_results = []
        
        for model_name, result in self.results.items():
            # Parse classification report into metrics
            report_lines = [line.strip() for line in result['report'].split('\n') if line.strip()]
            metrics = {}
            
            # Extract metrics from report
            for line in report_lines:
                if 'accuracy' in line:
                    continue  # We already have accuracy
                elif 'macro avg' in line:
                    parts = line.split()
                    metrics.update({
                        'precision': float(parts[-4]),
                        'recall': float(parts[-3]),
                        'f1': float(parts[-2])
                    })
                    break
            
            # Get model parameters
            model_params = {}
            try:
                model = result['pipeline'].named_steps['model']
                model_params = model.get_params()
                
                # Simplify some parameter names
                if hasattr(model, 'get_xgb_params'):
                    model_params.update(model.get_xgb_params())
                
                # Remove large objects that can't be serialized
                for k in list(model_params.keys()):
                    if callable(model_params[k]) or isinstance(model_params[k], (list, dict, np.ndarray)):
                        del model_params[k]
            except Exception as e:
                print(f"Could not get parameters for {model_name}: {str(e)}")
            
            # Build the result object
            model_result = {
                'name': model_name,
                'accuracy': float(result['accuracy']),
                'metrics': metrics,
                'report': result['report'],
                'trainingTime': float(result['train_time']),
                'cvTime': float(result['cv_time']),
                'cvScore': {
                    'mean': float(np.mean(result['cv_scores'])),
                    'std': float(np.std(result['cv_scores']))
                },
                'parameters': model_params,
                'description': self._get_model_description(model_name),
                'isBest': model_name == self.best_model_name
            }
            
            formatted_results.append(model_result)
        
        # Sort by accuracy (descending)
        formatted_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Mark the best model (should be first after sorting)
        if formatted_results:
            formatted_results[0]['isBest'] = True
            for m in formatted_results[1:]:
                m['isBest'] = False
        
        return formatted_results

    def _get_model_description(self, model_name):
        """Helper to get consistent model descriptions"""
        descriptions = {
            'LogisticRegression': 'Linear classifier with logistic function',
            'RandomForest': 'Ensemble of decision trees',
            'GradientBoosting': 'Boosting with gradient descent',
            'XGBoost': 'Extreme Gradient Boosting',
            'SVM': 'Support Vector Machine',
            'NaiveBayes': 'Probabilistic classifier',
            'DecisionTree': 'Non-linear decision boundaries',
            'KNN': 'Instance-based learning',
            'MLP': 'Artificial neural network',
            'LinearRegression': 'Ordinary least squares regression',
            'Ridge': 'L2-regularized linear regression',
            'Lasso': 'L1-regularized linear regression',
            'ElasticNet': 'Combined L1 and L2 regularization',
            'SVR': 'Support Vector Regression',
            'CatBoost': 'Gradient boosting with categorical support'
        }
        return descriptions.get(model_name, "Machine learning model")