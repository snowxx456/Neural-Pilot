from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from django.conf import settings
from django.http import FileResponse, HttpResponse
from .models import Dataset
import os
import json
import pandas as pd
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


logger = logging.getLogger(__name__)

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
def data_visualization(dataset_id):
    try:
        # Get dataset from database
        dataset = Dataset.objects.get(id=dataset_id)
        file_path = dataset.file.path
        df = pd.read_csv(file_path)
        # Perform data visualization here
    except Dataset.DoesNotExist:
        return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)

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
def model_training(request,id):
    try:
        dataset = Dataset.objects.get(id=id)
        file_path = dataset.file.path
        
        

        
        result = {
            "message": "Model training completed successfully",
            "model_path": "/path/to/saved/model"
        }

        return Response(result, status=status.HTTP_200_OK)
            
    except Exception as e:
        return Response({
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def select_dataset(request):
    try:
        dataset_ref = request.data.get('datasetRef')
        dataset_url = request.data.get('url')
        
        if not dataset_ref or not dataset_url:
            return Response({
                'error': 'Dataset reference and URL are required'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Create directory if it doesn't exist
        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Download file
        response = requests.get(dataset_url)
        response.raise_for_status()  # Raise exception for bad status codes
        
        file_path = os.path.join(dataset_dir, f'{dataset_ref}.csv')
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
            
        # Create Dataset model instance
        dataset = Dataset.objects.create(
            name=dataset_ref,
            file=file_path
        )
            
        return Response({
            'message': 'Dataset selected successfully',
            'dataset_id': dataset.id,
            'file_path': file_path
        }, status=status.HTTP_200_OK)
        
    except requests.RequestException as e:
        return Response({
            'error': f'Failed to download dataset: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return Response({
            'error': str(e)
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
