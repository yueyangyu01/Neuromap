from rest_framework import serializers
from .models import Example, Patient

class ExampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Example
        fields = (
            'name',
            'description',
        )
class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ['firstName', 'lastName', 'email', 'gender', 'dob', 'mri_image', 'diagnosis', 'affected_areas', 'treatment_options']

