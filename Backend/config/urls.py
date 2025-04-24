from django.urls import path
from api.views import health_check, upload_dataset, get_dataset, data_cleaning, search_dataset, model_training , select_dataset, download_dataset, sse_stream, start_preprocessing
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('api/health/', health_check),  # Note the trailing slash
    path('api/upload/', upload_dataset, name='upload_dataset'),  # Note the trailing slash
    path('api/dataset/<int:dataset_id>/', get_dataset),  # Note the trailing slash
    path('api/data_cleaning/<int:dataset_id>/', data_cleaning),  # Note the trailing slash
    path('api/search/',search_dataset),
    path('api/dataset/select/', select_dataset, name='select_dataset'),
    path('api/dataset/download/', download_dataset, name='download_dataset'),
    path('api/model_training/',model_training),  # Note the trailing slash
    path('api/sse-stream/', sse_stream, name='sse_stream'),
    path('api/start-preprocessing/', start_preprocessing, name='start_preprocessing'),  # Note the trailing slash
]

# Only enable this in development!
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)