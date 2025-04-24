from django.db import models
from django.utils import timezone

# Create your models here.
def dataset_upload_path(instance, filename):
    # Files will be stored in: uploads/datasets/2024/04/17/filename.csv
    return f'datasets/{timezone.now().strftime("%Y/%m/%d")}/{filename}'

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to=dataset_upload_path)  # Organized storage
    uploaded_at = models.DateTimeField(auto_now_add=True)

