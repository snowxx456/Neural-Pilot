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
from django.core.files import File


logger = logging.getLogger(__name__)

preprocessing_steps = {
    1: {"status": "pending"},
    2: {"status": "pending"},
    3: {"status": "pending"},
    4: {"status": "pending"},
    5: {"status": "pending"},
    6: {"status": "pending"},
}
import copy

def event_stream():
    """Generate SSE data"""
    last_sent = copy.deepcopy(preprocessing_steps)  # Changed to deepcopy
    
    while True:
        for step_id in preprocessing_steps:
            current = preprocessing_steps[step_id]
            last = last_sent[step_id]
            
            if current != last:  # Check if anything has changed (not just status)
                data = {
                    "id": step_id,
                    "status": current["status"]
                }
                
                # Add cleaned_dataset_id if it exists
                if "cleaned_dataset_id" in current:
                    data["cleaned_dataset_id"] = current["cleaned_dataset_id"]
                    data['status'] = "completed"  # Set status to completed for the last step
                    data["sample"] = current["sample"] if "sample" in current else None
                
                yield f"data: {json.dumps(data)}\n\n"
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

def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail
    raise ValueError(f"Could not read file {file_path} with any of the encodings: {encodings}")

# In views.py
def run_preprocessing_pipeline(id):
    
    
    update_step_status(1, "processing")
    # Step 1: Loading Dataset
    dataset = Dataset.objects.get(id=id)
    
    # Get the absolute file path
    import os
    from django.conf import settings
    
    # If using a relative media path stored in database
    if dataset.file.name.startswith('/'):
        # Remove leading slash if present
        file_path = dataset.file.name.lstrip('/')
    else:
        file_path = dataset.file.name
        
    # Get the full path
    full_path = os.path.join(settings.MEDIA_ROOT, file_path)
    
    # Print for debugging
    print(f"Looking for file at: {full_path}")
    
    # Check if file exists
    if not os.path.exists(full_path):
        print(f"File not found at: {full_path}")
        # Try the original path just in case
        if os.path.exists(dataset.file.url):
            full_path = dataset.file.url
            print(f"Found file at original path: {full_path}")
        else:
            raise FileNotFoundError(f"Could not find file at {full_path} or {dataset.file.url}")
    
    print(f"Dataset URL: {full_path}")
    df = read_csv_with_encoding(full_path)
    agent = CreateAgent(df)
    update_step_status(1, "completed")

    # Step 2: Handling Index Columns (NEW)
    update_step_status(2, "processing")
    agent.handle_index_column()
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
    # Save the cleaned dataset
    update_step_status(6, "processing")
    cleaned_dataset = agent.save_dataframe_to_csv(original_dataset_id=id)
    update_step_status(6, "completed")
    sample = agent.sample_data()
    
    # Send the cleaned dataset ID through SSE
    with preprocessing_steps_lock:
        preprocessing_steps[6] = {
            "status": "completed",
            "cleaned_dataset_id": cleaned_dataset.id,
            "sample":sample
        }
    

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
        
        # Get the absolute file path using the same approach as your download function
        if dataset.file.name.startswith('/'):
            file_path = dataset.file.name.lstrip('/')
        else:
            file_path = dataset.file.name
            
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"File not found at: {full_path}")
            return Response({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # Read the CSV file
        try:
            df = pd.read_csv(full_path)
            print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        except Exception as csv_error:
            print(f"CSV read error: {str(csv_error)}")
            return Response(
                {"error": f"Failed to read CSV file: {str(csv_error)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Convert DataFrame to dictionary for JSON serialization
        dataset_data = df.to_dict(orient='records')
        
        return Response(dataset_data, status=status.HTTP_200_OK)

    except Dataset.DoesNotExist:
        print(f"Dataset with ID {dataset_id} not found")
        return Response(
            {"error": "Dataset not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        print(f"Error sending dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response(
            {"error": f"Failed to process dataset: {str(e)}"}, 
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
def download_cleaned_dataset(request, id):
    try:
        # Get the dataset object
        dataset = Dataset.objects.get(id=id)
        
        # Get the absolute file path
        if dataset.file.name.startswith('/'):
            file_path = dataset.file.name.lstrip('/')
        else:
            file_path = dataset.file.name
            
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"File not found at: {full_path}")
            return Response({"error": "File not found"}, status=404)
        
        # For file downloads, we need to use Django's HttpResponse
        # We can't use DRF's Response for binary file downloads
        with open(full_path, 'rb') as file:
            response = HttpResponse(
                file.read(),
                content_type='text/csv'
            )
            
        # Set filename in Content-Disposition header
        filename = os.path.basename(full_path)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Dataset.DoesNotExist:
        print(f"Dataset with ID {id} not found")
        return Response({"error": "Dataset not found"}, status=404)
    except Exception as e:
        print(f"Download error: {str(e)}")
        return Response({"error": str(e)}, status=500)