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

class PreprocessingStep(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    step_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=100)
    description = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.step_id}: {self.title} ({self.status})"