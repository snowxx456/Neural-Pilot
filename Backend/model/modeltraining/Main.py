from library import *
from targetcolumn import TargetColumnRecommender
from data_load import DataLoader
from model_training import ModelTrainer
from visualization import VisualizationHandler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

target = TargetColumnRecommender(file_path="Nerual-Pilot\Backend\model\modeltraining\c.csv")
# Load the data and check for issues
if target.load_data():
    print("Data loaded successfully.")
    print("Performing data validation...")
    if target._validate_data():
        print("Data validation passed.")
        target.analyze_for_target()
        # target._is_potential_target()
        target_column = target.get_target_column()
        print(f"Recommended target column: {target.get_recommendation_reason()}")
        file_path = target.get_file_path()
        dataloader = DataLoader(file_path,target_column=target_column)
        if dataloader.load_data():
            print("Data loaded successfully.")
            dataloader._check_data_issues(dataloader.df)
            dataloader.set_target()
            x, y = dataloader.prepare_data()
            print(f"Data prepared: {x.shape[0]} rows, {x.shape[1]} features")
            df = dataloader.df
            print("Data preview:")
            print(df.head(3))
            print("Data info:")
            print(df.info())
            model_training = ModelTrainer(target_column=target_column, data=df, x=x, y=y, problem_type=dataloader.problem_type,preprocessor=dataloader.preprocessor)
            results, best_model, model_card =  model_training.train_models()
            print("Model training completed.")
            #hyper = model_training.hypertune_best_model()
            #print("Hyperparameter tuning completed.")
            print(model_training.results.keys())
            print("="*70)
            print("pro:",best_model)

            print("="*70)
            print(model_training.best_model)
            print("="*70)
            print(model_training.best_model_name)
            print("="*70)
            print(model_training.results[model_training.best_model_name])
            print("="*70)
            print(model_training.y_test)
            print("="*70)
            visualization = VisualizationHandler(data=df, best_model=model_training.best_model, best_model_name=model_training.best_model_name, results=results, 
                                                 feature_names=dataloader.feature_names, label_encoder=dataloader.label_encoder, 
                                                 probabilities=best_model, y_test=model_training.y_test, problem_type=dataloader.problem_type)
            print(visualization.correlation_graph())
            print("/n")
            print("="*70)
            print(visualization.feature_importance())
            print("/n")
            print("="*70)
            print(visualization.confusion_matrix())
            print("/n")
            print("="*70)
            print(visualization.roc_curve())
            print(visualization.precision_recall_curve())
            visualization.save_model(model_training.best_model_name)
            print(f"Model saved as {model_training.best_model_name}.pkl")
            print("Model saved successfully.")

    else:
        print("Data validation failed.")
else:
    print("Failed to load data. Please check the file path and format.")
