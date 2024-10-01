from rest_framework import serializers
import pandas as pd
from .models import CustomUser,Car



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



class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = CustomUser
        fields = ('email', 'name', 'phone_number', 'password')

    def create(self, validated_data):
        user = CustomUser.objects.create_user(**validated_data)
        return user

class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()


class CarSerializer(serializers.ModelSerializer):
    class Meta:
        model = Car
        fields = '__all__'