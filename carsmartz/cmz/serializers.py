from rest_framework import serializers
import pandas as pd
class CarDataSerializer(serializers.Serializer):
    company = serializers.CharField(max_length=100)  # Company name
    car_model = serializers.CharField(max_length=100)  # Car model
    year = serializers.IntegerField()  # Year of manufacture
    fuel_type = serializers.CharField(max_length=10)  # Fuel Type
    kilo_driven = serializers.FloatField()  # Kilometers driven

    def validate(self, data):
        # Optional: Add any additional validation logic here if needed
        # Example: Check if the year is reasonable
        current_year = pd.to_datetime("today").year
        if data['year'] > current_year:
            raise serializers.ValidationError("Year cannot be in the future.")
        return data
