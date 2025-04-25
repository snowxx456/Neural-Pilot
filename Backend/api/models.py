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

class ModelTrainingResult(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    problem_type = models.CharField(max_length=20)
    target_column = models.CharField(max_length=100)
    data = models.JSONField()
    best_model = models.JSONField()
    best_model_name = models.CharField(max_length=100)
    results = models.JSONField()
    feature_names = models.JSONField()
    label_encoder = models.JSONField(null=True)
    X_test = models.JSONField(null=True)
    y_test = models.JSONField(null=True)
    preprocessor = models.BinaryField(null=True)  # Store pickled preprocessor

    class Meta:
        get_latest_by = 'created_at'

