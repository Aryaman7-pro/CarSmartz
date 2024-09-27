import pickle
import pandas as pd
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from . serializers import CarDataSerializer
# Load the model and data
model = pickle.load(open('C:\\Users\\aryaman.kanwar\\CarSmartz\\StackedModel.pkl', 'rb'))
car = pd.read_csv('C:\\Users\\aryaman.kanwar\\CarSmartz\\Cleaned_Data.csv')


@api_view(['GET'])
def index(request):
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    # Create a dictionary for car models by company
    models_by_company = {company: [] for company in companies}
    for model in car_models:
        company_name = model.split(' ')[0]  # Assuming the company name is part of the model name
        if company_name in models_by_company:
            models_by_company[company_name].append(model)

    return render(request, 'index.html', {
        'companies': companies,
        'car_models': car_models,
        'models_by_company' : models_by_company,
        'years': years,
        'fuel_types': fuel_types
    })

@api_view(['POST'])
def predict(request):
    serializer = CarDataSerializer(data=request.data)
    
    if serializer.is_valid():
        validated_data = serializer.validated_data
        
        # Extract the validated data
        company = validated_data['company']
        car_model = validated_data['car_model']
        year = validated_data['year']
        fuel_type = validated_data['fuel_type']
        driven = validated_data['kilo_driven']

        # Prepare the input for the model
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                  data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))

        # Make the prediction
        prediction = model.predict(input_data)
        return Response({"prediction": np.round(prediction[0], 2)}, status=200)
    else:
        return Response(serializer.errors, status=400)
