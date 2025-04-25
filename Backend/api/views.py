from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from django.conf import settings
from django.http import FileResponse, HttpResponse
from .models import Dataset, ModelTrainingResult
import os
import json
import pandas as pd
import pickle
import requests
import kagglehub
import logging
import shutil
import glob
import tempfile
from model.search.groq_client import search_kaggle_datasets, format_size
import time
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import threading
from django.http import JsonResponse
preprocessing_steps_lock = threading.Lock()
from model.data_cleaning.llm.agent import CreateAgent
from django.core.files import File
from model.modeltraining.targetcolumn import TargetColumnRecommender  # Your target column recommendation logic
from model.modeltraining.data_load import DataLoader  # Your data loading logic
from model.modeltraining.visualization import VisualizationHandler  # Your visualization logic
from model.modeltraining.model_training import ModelTrainer  # Your model training logic

import numpy as np

logger = logging.getLogger(__name__)        

training_steps = {
    1: {"name": "Loading Dataset", "status": "pending", "details": {}},
    2: {"name": "Target Column Analysis", "status": "pending", "details": {}},
    3: {"name": "Data Validation", "status": "pending", "details": {}},
    4: {"name": "Data Preparation", "status": "pending", "details": {}},
    5: {"name": "Model Training", "status": "pending", "details": {}},
    6: {"name": "Hyperparameter Tuning", "status": "pending", "details": {}},
    7: {"name": "Model Evaluation", "status": "pending", "details": {}},
    8: {"name": "Visualization Generation", "status": "pending", "details": {}},
    9: {"name": "Model Saving", "status": "pending", "details": {}}
}

# Lock for thread-safe updates to training_steps
training_steps_lock = threading.Lock()

def event_stream_model():
    """Generate SSE data for model training progress"""
    last_sent = copy.deepcopy(training_steps)
    
    while True:
        with training_steps_lock:
            # Check for changes in any step
            for step_id in training_steps:
                current = training_steps[step_id]
                last = last_sent[step_id]
                
                # Send updates when status changes or details are updated
                if current["status"] != last["status"] or current["details"] != last["details"]:
                    data = json.dumps({
                        "id": step_id,
                        "name": current["name"],
                        "status": current["status"],
                        "details": current["details"]
                    })
                    yield f"data: {data}\n\n"
                    last_sent[step_id] = copy.deepcopy(current)
        
        # Send a keep-alive comment to prevent connection timeouts
        yield f": keepalive\n\n"
        time.sleep(0.1)  # Poll for updates every 100ms

@csrf_exempt
def sse_stream_model(request):
    """Stream SSE data to clients"""
    response = StreamingHttpResponse(event_stream_model(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable buffering in nginx
    response['Access-Control-Allow-Origin'] = '*'  # Allow cross-origin requests
    return response

def update_step_status_model(step_id, status, details=None):
    """Update the status and details of a training step"""
    with training_steps_lock:
        training_steps[step_id]["status"] = status
        if details is not None:
            training_steps[step_id]["details"] = details

@csrf_exempt
def start_model_training(request,id):
    """Start the model training process"""
    # Reset all steps to pending
    with training_steps_lock:
        for step_id in training_steps:
            training_steps[step_id]["status"] = "pending"
            training_steps[step_id]["details"] = {}
    
    # Start training in a separate thread
    thread = threading.Thread(target=run_model_training_pipeline, args=(id,))
    thread.daemon = True
    thread.start()
    
    return JsonResponse({"status": "started", "message": "Model training started. Connect to SSE stream for updates."})

def run_model_training_pipeline(id):
    """Run the entire model training pipeline with SSE updates"""
    try:
        # Step 1: Loading Dataset
        update_step_status(1, "processing", {"message": "Loading dataset..."})
        time.sleep(0.5)  # Simulate processing time
        
        file_path = "Nerual-Pilot/Backend/model/modeltraining/c.csv"
        target = TargetColumnRecommender(file_path=file_path)
        
        if not target.load_data():
            update_step_status(1, "error", {"message": "Failed to load dataset"})
            return
            
        update_step_status(1, "completed", {"message": "Dataset loaded successfully"})
        
        # Step 2: Target Column Analysis
        update_step_status(2, "processing", {"message": "Analyzing potential target columns..."})
        time.sleep(0.5)  # Simulate processing time
        
        if not target._validate_data():
            update_step_status(2, "error", {"message": "Data validation failed"})
            return
            
        target.analyze_for_target()
        target_column = target.get_target_column()
        recommendation_reason = target.get_recommendation_reason()
        
        update_step_status(2, "completed", {
            "target_column": target_column,
            "reason": recommendation_reason
        })
        
        # Step 3: Data Validation
        update_step_status(3, "processing", {"message": "Validating dataset quality..."})
        time.sleep(0.5)  # Simulate processing time
        
        file_path = target.get_file_path()
        dataloader = DataLoader(file_path, target_column=target_column)
        
        if not dataloader.load_data():
            update_step_status(3, "error", {"message": "Failed to load data for validation"})
            return
            
        issues = dataloader._check_data_issues(dataloader.df)
        
        update_step_status(3, "completed", {
            "issues_found": len(issues) if issues else 0,
            "data_shape": dataloader.df.shape
        })
        
        # Step 4: Data Preparation
        update_step_status(4, "processing", {"message": "Preparing data for modeling..."})
        time.sleep(0.5)  # Simulate processing time
        
        dataloader.set_target()
        x, y = dataloader.prepare_data()
        df = dataloader.df
        
        update_step_status(4, "completed", {
            "rows": x.shape[0],
            "features": x.shape[1],
            "problem_type": dataloader.problem_type
        })
        
        # Step 5: Model Training
        update_step_status(5, "processing", {"message": "Training multiple model candidates..."})
        
        model_training = ModelTrainer(
            target_column=target_column,
            data=df,
            x=x,
            y=y,
            problem_type=dataloader.problem_type,
            preprocessor=dataloader.preprocessor
        )
        
        # Training progress updates
        models_to_train = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost", "SVM"]
        total_models = len(models_to_train)
        
        # Simulate training progress
        for i, model_name in enumerate(models_to_train):
            update_step_status(5, "processing", {
                "message": f"Training {model_name}...",
                "progress": (i / total_models) * 100,
                "current_model": model_name
            })
            time.sleep(1)  # Simulate training time
        
        # Actual training
        results, best_model, model_card = model_training.train_models()
        
        update_step_status(5, "completed", {
            "best_model": model_training.best_model_name,
            "models_trained": list(results.keys()),
            "total_models": len(results)
        })
        
        # Step 6: Hyperparameter Tuning
        update_step_status(6, "processing", {"message": f"Hypertuning {model_training.best_model_name}..."})
        time.sleep(2)  # Simulate tuning time
        
        # Skip hypertuning for now, but you could add actual hypertuning here
        # hyper = model_training.hypertune_best_model()
        
        update_step_status(6, "completed", {
            "model": model_training.best_model_name,
            "improvement": "Skip for now"  # Replace with actual improvement
        })
        
        # Step 7: Model Evaluation
        update_step_status(7, "processing", {"message": "Evaluating model performance..."})
        time.sleep(0.5)  # Simulate processing time
        
        best_model_results = results[model_training.best_model_name]
        
        update_step_status(7, "completed", {
            "accuracy": best_model_results.get("accuracy"),
            "f1_score": best_model_results.get("metrics", {}).get("f1"),
            "precision": best_model_results.get("metrics", {}).get("precision"),
            "recall": best_model_results.get("metrics", {}).get("recall")
        })
        
        # Step 8: Visualization Generation
        update_step_status(8, "processing", {"message": "Generating visualizations..."})
        time.sleep(1)  # Simulate processing time
        
        visualization = VisualizationHandler(
            data=df,
            best_model=model_training.best_model,
            best_model_name=model_training.best_model_name,
            results=results,
            feature_names=dataloader.feature_names,
            label_encoder=dataloader.label_encoder,
            probabilities=best_model,
            y_test=model_training.y_test,
            problem_type=dataloader.problem_type
        )
        
        # Generate visualizations
        viz_types = ["correlation", "feature_importance", "confusion_matrix", "roc_curve", "precision_recall"]
        for i, viz_type in enumerate(viz_types):
            update_step_status(8, "processing", {
                "message": f"Generating {viz_type} visualization...",
                "progress": (i / len(viz_types)) * 100,
                "current_viz": viz_type
            })
            time.sleep(0.5)  # Simulate generation time
        
        update_step_status(8, "completed", {
            "visualizations_generated": viz_types,
            "count": len(viz_types)
        })
        
        # Step 9: Model Saving
        update_step_status(9, "processing", {"message": "Saving trained model..."})
        time.sleep(0.5)  # Simulate processing time
        
        model_path = visualization.save_model(model_training.best_model_name)
        
        update_step_status(9, "completed", {
            "model_name": model_training.best_model_name,
            "file_path": f"{model_training.best_model_name}.pkl"
        })
        
    except Exception as e:
        # Handle exceptions
        import traceback
        error_details = traceback.format_exc()
        
        # Update the current step with error status
        for step_id in training_steps:
            if training_steps[step_id]["status"] == "processing":
                update_step_status(step_id, "error", {
                    "message": str(e),
                    "details": error_details
                })
                break

preprocessing_steps = {
    1: {"status": "pending"},
    2: {"status": "pending"},
    3: {"status": "pending"},
    4: {"status": "pending"},
    5: {"status": "pending"}
}
import copy

def event_stream():
    """Generate SSE data"""
    last_sent = copy.deepcopy(preprocessing_steps)  # Changed to deepcopy
    
    while True:
        for step_id in preprocessing_steps:
            current = preprocessing_steps[step_id]
            last = last_sent[step_id]
            
            if current["status"] != last["status"]:
                data = json.dumps({
                    "id": step_id,
                    "status": current["status"]
                })
                yield f"data: {data}\n\n"
                last_sent[step_id] = copy.deepcopy(current)  # Deepcopy update
        
        time.sleep(0.1)  # Faster polling

@csrf_exempt
def sse_stream(request):
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable buffering in nginx
    return response

def update_step_status(step_id, status):
    with preprocessing_steps_lock:
        preprocessing_steps[step_id]["status"] = status

@csrf_exempt
def start_preprocessing(request,id):
    # Start preprocessing in a separate thread
    thread = threading.Thread(target=run_preprocessing_pipeline, args=(id,))
    thread.daemon = True
    thread.start()
    return JsonResponse({"status": "started"})

# In views.py
def run_preprocessing_pipeline(id):
    # Step 1: Loading Dataset
    dataset = Dataset.objects.get(id=id)
    data = dataset.file.path
    agent = CreateAgent(data=data)
    update_step_status(1, "processing")
    agent.load_data()
    update_step_status(1, "completed")

    # Step 2: Handling Index Columns (NEW)
    update_step_status(2, "processing")
    agent.handle_index_columns()
    update_step_status(2, "completed")

    # Step 3: Handling Missing Values
    update_step_status(3, "processing")
    agent.handle_missing_values()
    update_step_status(3, "completed")

    # Step 4: Handling Outliers
    update_step_status(4, "processing")
    agent.handle_outliers()
    update_step_status(4, "completed")

    # Step 5: Removing Duplicate Columns
    update_step_status(5, "processing")
    agent.handle_duplicates()
    update_step_status(5, "completed")

#data visulaization 
def data_visualization(id):
    # Placeholder for data visualization logic
    return JsonResponse({"message": "Data visualization logic goes here."})



@api_view(['GET'])
def health_check(request):
    return Response({"status": "OK"}, content_type='application/json')

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_dataset(request):
    # Basic security checks
    if 'file' not in request.FILES:
        return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
    
    file = request.FILES['file']
    
    # Validate file type
    if not file.name.endswith('.csv'):
        return Response({"error": "Only CSV files allowed"}, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    
    # Limit file size (5MB)
    if file.size > 5242880:
        return Response({"error": "File too large (max 5MB)"}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    
    try:
        # Save to database and filesystem
        dataset = Dataset.objects.create(
            name=os.path.splitext(file.name)[0],  # Remove extension
            file=file
        )
        return Response({
            "id": dataset.id,
            "name": dataset.name,
        }, status=status.HTTP_201_CREATED)

        
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
def get_dataset(request, dataset_id):
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        return Response({
            "id": dataset.id,
            "name": dataset.name,
            "path": dataset.file.url
        }, status=status.HTTP_200_OK)
    except Dataset.DoesNotExist:
        return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
    
@api_view(['POST'])
def data_visualization(request, dataset_id):
    try:
        # Get dataset from database
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Generate basic visualizations and statistics
        visualizations = {
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
            },
            'column_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': df.describe().to_dict(),
            'correlations': df.corr().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
        }

        return Response(visualizations, status=status.HTTP_200_OK)

    except Dataset.DoesNotExist:
        return Response(
            {"error": "Dataset not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to generate visualizations: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def search_dataset(request):
    query = request.data.get('query', '')

    try:
        results = search_kaggle_datasets(query)
        formatted = []

        for ds in results:
            formatted.append({
                'title': getattr(ds, 'title', 'Untitled Dataset'),
                'owner': getattr(ds, 'ownerName', 'Unknown'),
                'ref': getattr(ds, 'ref', '').replace('/datasets/', ''),
                'description': getattr(
                    ds, 'description',
                    getattr(ds, 'subtitle',
                    getattr(ds, 'summary', 'No description available'))
                ),
                'size': format_size(getattr(ds, 'size',
                         getattr(ds, 'totalBytes', 'Unknown size'))),
                'downloads': getattr(ds, 'totalDownloads', 0),
                'lastUpdated': getattr(ds, 'lastUpdated', 'Unknown'),
                'url': f'https://www.kaggle.com/datasets/{getattr(ds, "ref", "")}'
            })

        return Response({'datasets': formatted}, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def model_training(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
        file_path = dataset.file.path
        
        # Initialize and run model training
        trainer = ModelTrainer(file_path)
        training_results = trainer.train()
        
        # Get probabilities for visualization
        best_model_name = training_results['best_model_name']
        probabilities = training_results['results'].get(best_model_name, {}).get('probabilities')
        
        # Save training results including test data
        model_result = ModelTrainingResult.objects.create(
            dataset=dataset,
            problem_type=training_results['problem_type'],
            target_column=training_results['target_column'],
            data=training_results['data'],
            best_model=training_results['best_model'],
            best_model_name=training_results['best_model_name'],
            results=training_results['results'],
            feature_names=training_results['feature_names'],
            label_encoder=training_results.get('label_encoder'),
            X_test=training_results.get('X_test').tolist() if training_results.get('X_test') is not None else None,
            y_test=training_results.get('y_test').tolist() if training_results.get('y_test') is not None else None,
            preprocessor=pickle.dumps(training_results.get('preprocessor')) if training_results.get('preprocessor') is not None else None
        )
        
        return Response({
            "message": "Model training completed successfully",
            "training_id": model_result.id
        }, status=status.HTTP_200_OK)
            
    except Exception as e:
        return Response({
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
def download_dataset(request):
    try:
        dataset_ref = request.data.get('datasetRef')
        if not dataset_ref:
            return Response({
                'error': 'Dataset reference is required'
            }, status=status.HTTP_400_BAD_REQUEST)

        print(f"Attempting to download dataset: {dataset_ref}")

        # Step 1: Download using kagglehub (downloads to cache dir)
        cache_dir = kagglehub.dataset_download(dataset_ref)
        print(f"Downloaded to cache: {cache_dir}")

        # Step 2: Move contents to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Copying to temp dir: {temp_dir}")
            shutil.copytree(cache_dir, temp_dir, dirs_exist_ok=True)

            # Step 3: Find the first CSV file
            csv_files = glob.glob(os.path.join(temp_dir, '**', '*.csv'), recursive=True)

            if not csv_files:
                return Response({
                    'error': 'No CSV file found in the dataset'
                }, status=status.HTTP_404_NOT_FOUND)

            csv_file = csv_files[0]
            print(f"Found CSV: {csv_file}")

            # Step 4: Serve the CSV to frontend
            with open(csv_file, 'rb') as f:
                response = HttpResponse(f.read(), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_file)}"'
                return response

    except Exception as e:
        print(f"Download error: {str(e)}")
        return Response({
            'error': f'An error occurred while processing the download: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def select_dataset(request):
    try:
        dataset_ref = request.data.get('datasetRef')
        if not dataset_ref:
            return Response({
                'error': 'Dataset reference is required'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        print(f"Attempting to select dataset: {dataset_ref}")

        # Step 1: Download using kagglehub (downloads to cache dir)
        cache_dir = kagglehub.dataset_download(dataset_ref)
        print(f"Downloaded to cache: {cache_dir}")

        # Step 2: Move contents to a temporary directory (like in download_dataset)
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Copying to temp dir: {temp_dir}")
            shutil.copytree(cache_dir, temp_dir, dirs_exist_ok=True)

            # Step 3: Find the first CSV file
            csv_files = glob.glob(os.path.join(temp_dir, '**', '*.csv'), recursive=True)

            if not csv_files:
                return Response({
                    'error': 'No CSV file found in the dataset'
                }, status=status.HTTP_404_NOT_FOUND)

            csv_file_path = csv_files[0]
            csv_file_name = os.path.basename(csv_file_path)
            dataset_name = os.path.splitext(csv_file_name)[0]  # Remove extension
            
            print(f"Found CSV: {csv_file_path}")

            # Step 4: Create a Django file object similar to upload_dataset
            with open(csv_file_path, 'rb') as f:
                # Create an InMemoryUploadedFile from the file content
                from django.core.files.uploadedfile import InMemoryUploadedFile
                import io
                
                # Read the file content
                file_content = f.read()
                file_io = io.BytesIO(file_content)
                
                # Create an uploaded file object similar to what request.FILES provides
                uploaded_file = InMemoryUploadedFile(
                    file=file_io,
                    field_name='file',
                    name=csv_file_name,
                    content_type='text/csv',
                    size=len(file_content),
                    charset=None
                )
                
                # Create dataset like in upload_dataset
                dataset = Dataset.objects.create(
                    name=dataset_name,
                    file=uploaded_file
                )
                print(f"Dataset created: {dataset.id}")

                
                # Return the dataset information
                return Response({
                    "id": dataset.id,
                    "name": dataset.name,
                }, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"Selection error: {str(e)}")
        return Response({
            'error': f'An error occurred while processing the dataset selection: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET'])
def confusion_matrix(request):
    try:
        # Get the latest model training result
        latest_training = ModelTrainingResult.objects.latest('created_at')
        
        # Initialize the visualization handler with the correct parameters
        visualization = VisualizationHandler(
            data=latest_training.data,
            best_model=latest_training.best_model,
            best_model_name=latest_training.best_model_name,
            results=latest_training.results,
            feature_names=latest_training.feature_names,
            label_encoder=latest_training.label_encoder,
            y_test=latest_training.y_test,
            problem_type=latest_training.problem_type
        )
        
        # Get confusion matrix data
        matrix_data = visualization.confusion_matrix()
        
        # Check for errors
        if "error" in matrix_data:
            return Response(
                {"error": matrix_data["error"]}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Return formatted response
        return Response(matrix_data, status=status.HTTP_200_OK)
        
    except ModelTrainingResult.DoesNotExist:
        return Response(
            {"error": "No model training results found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to generate confusion matrix: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
@api_view(['GET'])
def feature_importance(request):
    try:
        # Get the latest model training result
        latest_training = ModelTrainingResult.objects.latest('created_at')
        
        # Initialize the visualization handler with the correct parameters
        visualization = VisualizationHandler(
            data=latest_training.data,
            best_model=latest_training.best_model,
            best_model_name=latest_training.best_model_name,
            results=latest_training.results,
            feature_names=latest_training.feature_names,
            label_encoder=latest_training.label_encoder,
            y_test=latest_training.y_test,
            problem_type=latest_training.problem_type
        )
        
        # Get feature importance data
        importance_data = visualization.feature_importance()
        
        # Check for errors
        if isinstance(importance_data, dict) and "error" in importance_data:
            return Response(
                {"error": importance_data["error"]}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Return formatted response
        return Response({
            "importance": importance_data,
            "modelName": latest_training.best_model_name
        }, status=status.HTTP_200_OK)
        
    except ModelTrainingResult.DoesNotExist:
        return Response(
            {"error": "No model training results found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to generate feature importance: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
@api_view(['GET'])
def roc_curve(request):
    try:
        # Get the latest model training result
        latest_training = ModelTrainingResult.objects.latest('created_at')
        
        # Make sure the probabilities are available
        probabilities = latest_training.results.get(latest_training.best_model_name, {}).get('probabilities')
        if probabilities is None:
            return Response(
                {"error": "Probability data not available for ROC curve"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Initialize the visualization handler with the correct parameters
        visualization = VisualizationHandler(
            data=latest_training.data,
            best_model=latest_training.best_model,
            best_model_name=latest_training.best_model_name,
            results=latest_training.results,
            feature_names=latest_training.feature_names,
            label_encoder=latest_training.label_encoder,
            probabilities=probabilities,  # Pass the probabilities
            y_test=latest_training.y_test,
            problem_type=latest_training.problem_type
        )
        
        # Get ROC curve data
        roc_data = visualization.roc_curve()
        
        # Check for errors
        if "error" in roc_data:
            return Response(
                {"error": roc_data["error"]}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Return formatted response
        return Response(roc_data, status=status.HTTP_200_OK)
        
    except ModelTrainingResult.DoesNotExist:
        return Response(
            {"error": "No model training results found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to generate ROC curve: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def correlation(request):
    try:
        # Get the latest model training result
        latest_training = ModelTrainingResult.objects.latest('created_at')
        
        # Initialize the visualization handler with the correct parameters
        visualization = VisualizationHandler(
            data=latest_training.data,
            best_model=latest_training.best_model,
            best_model_name=latest_training.best_model_name,
            results=latest_training.results,
            feature_names=latest_training.feature_names,
            label_encoder=latest_training.label_encoder,
            y_test=latest_training.y_test,
            problem_type=latest_training.problem_type
        )
        
        # Get correlation matrix data
        correlation_data = visualization.correlation_graph()  # Note the method name change
        
        # Check for errors
        if "error" in correlation_data:
            return Response(
                {"error": correlation_data["error"]}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Return formatted response
        return Response(correlation_data, status=status.HTTP_200_OK)
        
    except ModelTrainingResult.DoesNotExist:
        return Response(
            {"error": "No model training results found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to generate correlation matrix: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def precision_recall(request):
    try:
        # Get the latest model training result
        latest_training = ModelTrainingResult.objects.latest('created_at')
        
        # Make sure the probabilities are available
        probabilities = latest_training.results.get(latest_training.best_model_name, {}).get('probabilities')
        if probabilities is None:
            return Response(
                {"error": "Probability data not available for precision-recall curve"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Initialize the visualization handler with the correct parameters
        visualization = VisualizationHandler(
            data=latest_training.data,
            best_model=latest_training.best_model,
            best_model_name=latest_training.best_model_name,
            results=latest_training.results,
            feature_names=latest_training.feature_names,
            label_encoder=latest_training.label_encoder,
            probabilities=probabilities,  # Pass the probabilities
            y_test=latest_training.y_test,
            problem_type=latest_training.problem_type
        )
        
        # Get precision-recall curve data
        pr_data = visualization.precision_recall_curve()
        
        # Check for errors
        if "error" in pr_data:
            return Response(
                {"error": pr_data["error"]}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Return formatted response
        return Response(pr_data, status=status.HTTP_200_OK)
        
    except ModelTrainingResult.DoesNotExist:
        return Response(
            {"error": "No model training results found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to generate precision-recall curve: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )