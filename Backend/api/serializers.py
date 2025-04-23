from rest_framework import serializers
from .models import PreprocessingStep

class PreprocessingStepSerializer(serializers.ModelSerializer):
    class Meta:
        model = PreprocessingStep
        fields = ['step_id', 'title', 'description', 'status']
        # If you want to rename 'step_id' to just 'id' in the API output:
        extra_kwargs = {
            'step_id': {'source': 'id'}
        }

# Optional: Add more specialized serializers as needed
class PreprocessingStepStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = PreprocessingStep
        fields = ['step_id', 'status']