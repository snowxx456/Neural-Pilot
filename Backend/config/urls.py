from django.urls import path
from api.views import correlation, precision_recall, roc_curve, feature_importance, confusion_matrix, health_check, upload_dataset, get_dataset, search_dataset, model_training , select_dataset, download_dataset, sse_stream, start_preprocessing, select_dataset,data_visualization,start_model_training,sse_stream_model, download_cleaned_dataset
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('api/health/', health_check),  # Note the trailing slash
    path('api/upload/', upload_dataset, name='upload_dataset'),  # Note the trailing slash
    path('api/dataset/<int:dataset_id>/', get_dataset),  # Note the trailing slash
    path('api/search/',search_dataset),
    path('api/dataset/select/', select_dataset, name='select_dataset'),
    path('api/dataset/download/', download_dataset, name='download_dataset'),
    path('api/model_training/',model_training),  # Note the trailing slash
    path('api/sse-stream/', sse_stream, name='sse_stream'),
    path('api/start-preprocessing/<int:id>/', start_preprocessing, name='start_preprocessing'),  # Note the trailing slash
    path('api/dataset/select/', select_dataset, name='select_dataset'),  # Note the trailing slash
    path('api/visualization/<int:dataset_id>/', data_visualization, name='data_visualization'),
    path('api/start-preprocessing/', start_preprocessing, name='start_preprocessing'),  # Note the trailing slash
    path('api/dataset/select/', select_dataset, name='select_dataset'),  
    path('api/download-cleaned-dataset/<int:id>/', download_cleaned_dataset, name='download_cleaned_dataset'),  # Note the trailing 
    path('api/confusionmatrix/', confusion_matrix, name='confusion_matrix'),
    path('api/featureimportance/', feature_importance, name='feature_importance'),
    path('api/roc_curve/', roc_curve, name='roc_curve'),
    path('api/correlationmatrix/', correlation, name='correlation'),
    path('api/precision_recall/', precision_recall, name='precision_recall'),
    path('api/stream/', sse_stream_model, name='sse_stream_model'),  # Note the trailing slash
    path('api/train/<int:id>/', start_model_training,name="start_model_training"),  # Note the trailing slash
]

# Only enable this in development!
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)