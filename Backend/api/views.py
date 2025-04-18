from rest_framework.decorators import api_view , parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from .models import Dataset
import os
from model.data_cleaning.llm_cleaning import LLMCLEANINGAGENT
import tempfile
import json
import pandas as pd
from django.conf import settings
from model.search.groq_client import search_kaggle_datasets,format_size
import logging
from model.modeltraining.model_training import training

logger = logging.getLogger(__name__)

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
            "path": dataset.file.url
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
    
@api_view(['GET'])
def data_cleaning(request, dataset_id):
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        
        # Create cleaned file path using MEDIA_ROOT
        cleaned_file_name = f"cleaned_{os.path.basename(dataset.file.name)}"
        cleaned_file_path = os.path.join(settings.MEDIA_ROOT, 'cleaned_datasets', cleaned_file_name)

        # Create directory structure if needed
        os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)
                
        # Get the full path of the file
        file_path = dataset.file.path
        
        # Create LLM cleaning agent
        cleaning_agent = LLMCLEANINGAGENT(
            input_file=file_path,  # Use the file path
            llm_model="deepseek-r1-distill-llama-70b",  # You can make this configurable
            cleaning_strategy="auto"
        )
        
        # Run the cleaning pipeline
        cleaning_result = cleaning_agent.run()
        
        if not cleaning_result:
            return Response({"error": "Data cleaning failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Save the cleaned dataframe to the new file
        cleaning_result["cleaned_df"].to_csv(cleaned_file_path, index=False)
        
        # Create a results object
        results = {
            "message": "Data cleaning successful",
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "original_path": dataset.file.url,
                "cleaned_path": f"/media/cleaned_{os.path.basename(dataset.file.name)}"
            },
            "cleaning_stats": {
                "original_rows": len(cleaning_agent.df),
                "cleaned_rows": len(cleaning_result["cleaned_df"]),
                "missing_values_before": int(cleaning_agent.df.isnull().sum().sum()),
                "missing_values_after": int(cleaning_result["cleaned_df"].isnull().sum().sum())
            }
        }
        
        
        # Save suggestions to a file
        suggestions_dir = os.path.join(settings.MEDIA_ROOT, 'suggestions')
        os.makedirs(suggestions_dir, exist_ok=True)  # Create directory if needed
        suggestions_path = os.path.join(suggestions_dir, f'suggestions_{dataset_id}.json')

        # Save cleaning code to a file 
        code_dir = os.path.join(settings.MEDIA_ROOT, 'cleaning_codes')
        os.makedirs(code_dir, exist_ok=True)
        code_path = os.path.join(code_dir, f'cleaning_code_{dataset_id}.py')

        # Write files
        with open(suggestions_path, 'w') as f:
            # Convert suggestions to JSON string if needed
            if isinstance(cleaning_result["suggestions"], dict):
                json.dump(cleaning_result["suggestions"], f, indent=2)
            else:
                f.write(cleaning_result["suggestions"])

        with open(code_path, 'w') as f:
            f.write(cleaning_result["cleaning_code"])

        # Update response paths to use MEDIA_URL
        results = {
            "cleaning_suggestions_path": os.path.join(settings.MEDIA_URL, 'suggestions', f'suggestions_{dataset_id}.json'),
            "cleaning_code_path": os.path.join(settings.MEDIA_URL, 'cleaning_codes', f'cleaning_code_{dataset_id}.py'),
            }
        
        return Response(results, status=status.HTTP_200_OK)
        
    except Dataset.DoesNotExist:
        return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
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

@api_view(["POST"])
def model_training(request):
    file_path = request.data.get('file_path','')
    target_column = request.data.get('target_column','')
    try:
        # Load the dataset from the file path
        training()
    except:
        return Response({"error": "Failed to train model"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)