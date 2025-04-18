from django.urls import path
from api.views import health_check, upload_dataset, get_dataset, data_cleaning, search_dataset
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('api/health/', health_check),  # Note the trailing slash
    path('api/upload/', upload_dataset, name='upload_dataset'),  # Note the trailing slash
    path('api/dataset/<int:dataset_id>/', get_dataset),  # Note the trailing slash
    path('api/data_cleaning/<int:dataset_id>/', data_cleaning),  # Note the trailing slash
    path('api/search/',search_dataset)
]

# Only enable this in development!
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)