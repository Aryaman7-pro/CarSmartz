import pickle
import pandas as pd
import numpy as np
from cmz.models import Car
import plotly.graph_objects as go
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from django.shortcuts import render
from . serializers import CarDataSerializer, UserRegistrationSerializer, UserLoginSerializer, CarSerializer
from .models import Car
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken
from django.contrib.auth import authenticate
from django.shortcuts import redirect
import matplotlib.pyplot as plt
from rest_framework_simplejwt.views import TokenRefreshView
import io
import base64
import logging
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.pagination import PageNumberPagination
from django.conf import settings
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.contrib.auth import get_user_model
from django.contrib.auth import logout
User = get_user_model()
# Load the model and data
model = pickle.load(open('C:\\Users\\aryaman.kanwar\\CarSmartz\\StackedModel.pkl', 'rb'))
car_pipeline = pickle.load(open('C:\\Users\\aryaman.kanwar\\CarSmartz\\car_recommendation_model.pkl', 'rb'))
car = pd.read_csv('C:\\Users\\aryaman.kanwar\\CarSmartz\\Cleaned_Data.csv')
df = pd.read_csv('C:\\Users\\aryaman.kanwar\\CarSmartz\\Recommendation_Data.csv')
logger = logging.getLogger(__name__)

class SmallPageNumberPagination(PageNumberPagination):
    page_size = 12  # Set default page size to 3
    page_size_query_param = 'page_size'
    max_page_size = 10


def get_valid_access_token_user(request):
    access_token = request.COOKIES.get('access')
    refresh_token = request.COOKIES.get('refresh')

    if not access_token:
        raise AuthenticationFailed('Access token is required')

    try:
        # Decode the access token to get the user's first name and last name
        decoded_access_token = AccessToken(access_token)
        email = decoded_access_token.get('email')  # Adjust the key based on your token's payload   # Adjust the key based on your token's payload

        # Check if the user with the given first name and last name exists in the HeadTeamModel
        if not User.objects.filter(email=email).exists():
            raise AuthenticationFailed('User is not authorized')

        return access_token

    except Exception as e:
        # If access token validation fails, check the refresh token
        if not refresh_token:
            raise AuthenticationFailed('Refresh token is required')

        try:
            refresh = RefreshToken(refresh_token)
            new_access_token = str(refresh.access_token)

            # Verify if the new access token corresponds to a valid HeadTeam user
            decoded_new_access_token = AccessToken(new_access_token)
            email = decoded_access_token.get('email')    # Adjust the key based on your token's payload

            if not User.objects.filter(email=email).exists():
                raise AuthenticationFailed('User is not authorized')

            return new_access_token

        except Exception as e:
            raise AuthenticationFailed('Invalid refresh token')


@api_view(['GET'])
def index(request):
    # try:
    #     access_token = get_valid_access_token_user(request)
        
    #     if access_token != request.COOKIES.get('access'):
    #         response = Response()
    #         response.set_cookie(
    #             'access', access_token,
    #             httponly=True,
    #             secure=True,
    #             samesite='Strict'
    #         )
    #     else:
    #         response = Response()

    # except AuthenticationFailed:
    #     return Response({'error': "Unauthorized access. Token is invalid or expired."}, status=401)
    try:
        access_token = get_valid_access_token_user(request)
        response = Response({"message": "Your view's response data"})

        if access_token != request.COOKIES.get('access'):
            response.set_cookie(
                'access', access_token,
                httponly=True,
                secure=True,
                samesite='Strict',
                max_age=['ACCESS_TOKEN_LIFETIME'].seconds  
            )
    except AuthenticationFailed:
        return Response({'error': "Unauthorized access. Token is invalid or expired."}, status=401)
    
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
    try:
        access_token = get_valid_access_token_user(request)
        response = Response({"message": "Your view's response data"})

        if access_token != request.COOKIES.get('access'):
            response.set_cookie(
                'access', access_token,
                httponly=True,
                secure=True,
                samesite='Strict',
                max_age=['ACCESS_TOKEN_LIFETIME'].seconds  
            )
    except AuthenticationFailed:
        return Response({'error': "Unauthorized access. Token is invalid or expired."}, status=401)
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




@api_view(['GET'])
def depreciation_graph(request):
    try:
        access_token = get_valid_access_token_user(request)
        response = Response({"message": "Your view's response data"})

        if access_token != request.COOKIES.get('access'):
            response.set_cookie(
                'access', access_token,
                httponly=True,
                secure=True,
                samesite='Strict',
                max_age=['ACCESS_TOKEN_LIFETIME'].seconds  
            )
    except AuthenticationFailed:
        return Response({'error': "Unauthorized access. Token is invalid or expired."}, status=401)
    if request.method == 'GET':
        logger.info(f"Received request: {request.GET}")
        logger.info(f"Headers: {request.headers}")

        # Check if it's an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            logger.info("Processing AJAX request")
            # This is an AJAX request, process it as before
            initial_price = request.query_params.get('initial_price')
            num_months = request.query_params.get('num_months')
            
            logger.info(f"Received parameters: initial_price={initial_price}, num_months={num_months}")

            # Validate input
            try:
                initial_price = float(initial_price)
                num_months = int(num_months)
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid input: {str(e)}")
                return Response(
                    {"error": "Invalid input. Please provide valid initial_price and num_months."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if initial_price <= 0 or num_months <= 0:
                logger.error("Invalid input: Values must be positive")
                return Response(
                    {"error": "initial_price and num_months must be positive."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Generate data
            months = np.arange(0, num_months)
            
            # Depreciation function (exponential decay)
            def depreciation(t, initial_price, rate=0.005):
                return initial_price * np.exp(-rate * t)
            
            prices = depreciation(months, initial_price)
            
            # Create the plot
            plt.figure(figsize=(8, 4))
            plt.plot(months, prices)
            plt.title(f'General Car Price Depreciation Over {num_months} Months')
            plt.xlabel('Months')
            plt.ylabel('Price (Rs)')
            plt.grid(True)
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Clear the current figure
            plt.clf()
            
            logger.info("Successfully generated graph")
            return Response({
                'image': image,
                'data': {
                    'months': months.tolist(),
                    'prices': prices.tolist()
                }
            })
        else:
            logger.info("Rendering HTML template")
            # This is a regular GET request, render the HTML template
            return render(request, 'home.html')



@api_view(['GET','POST'])
def user_registration(request):
    if request.method == 'POST':
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    return render(request, 'register.html')



@api_view(['GET', 'POST'])
def user_login(request):
    if request.method == 'POST':
        email = request.data.get('email')
        password = request.data.get('password')
        
        logger.debug(f"Login attempt for email: {email}")

        try:
            user = User.objects.get(email=email)
            logger.debug(f"User found: {user}")
        except User.DoesNotExist:
            logger.warning(f"User with email {email} does not exist")
            return Response({'error': 'User does not exist'}, status=status.HTTP_401_UNAUTHORIZED)
        
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            logger.info(f"User {email} authenticated successfully")
            refresh = RefreshToken.for_user(user)
            logger.debug(f"Refresh token generated: {refresh}")
            logger.debug(f"Access token generated: {refresh.access_token}")

            response = Response({
                'message': 'Login successful',
                'user_id': user.id,
                'email': user.email
            }, status=status.HTTP_200_OK)

            # Set cookies
            response.set_cookie(
                'access',
                str(refresh.access_token),
                max_age=settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME'].total_seconds(),
                httponly=True,
                samesite='Lax',
                path='/'
            )
            response.set_cookie(
                'refresh',
                str(refresh),
                max_age=settings.SIMPLE_JWT['REFRESH_TOKEN_LIFETIME'].total_seconds(),
                httponly=True,
                samesite='Lax',
                path='/'
            )

            return response
             
        else:
            logger.warning(f"Authentication failed for user {email}")
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
    
    return render(request, 'login.html')


def logout_view(request):
    logout(request)  # Log out the user
    response = redirect('login')  # Redirect to the login page or home page
    response.delete_cookie('access', path='/')  # Clear the access token cookie
    response.delete_cookie('refresh', path='/')  # Clear the refresh token cookie
    return response


@api_view(['POST'])
def create_car(request):
    try:
        access_token = get_valid_access_token_user(request)
        response = Response({"message": "Your view's response data"})

        if access_token != request.COOKIES.get('access'):
            response.set_cookie(
                'access', access_token,
                httponly=True,
                secure=True,
                samesite='Strict',
                max_age=['ACCESS_TOKEN_LIFETIME'].seconds  
            )
    except AuthenticationFailed:
        return Response({'error': "Unauthorized access. Token is invalid or expired."}, status=401)
    serializer = CarSerializer(data=request.data)
    print(serializer)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    else:
        print(serializer.errors)  # Add this line to debug
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    


@api_view(['GET'])
def get_all_cars(request):
        try:
            access_token = get_valid_access_token_user(request)
            response = Response({"message": "Your view's response data"})

            if access_token != request.COOKIES.get('access'):
               response.set_cookie(
                'access', access_token,
                httponly=True,
                secure=True,
                samesite='Strict',
                max_age=['ACCESS_TOKEN_LIFETIME'].seconds  
            )
        except AuthenticationFailed:
              return Response({'error': "Unauthorized access. Token is invalid or expired."}, status=401)
        cars = Car.objects.all()  # Fetch all car records

        # Apply pagination
        paginator = SmallPageNumberPagination()
        paginated_cars = paginator.paginate_queryset(cars, request)

        serializer = CarSerializer(paginated_cars, many=True)

        # Render the paginated data in the HTML template
        return render(request, 'buy.html', {
            'cars': serializer.data,
            'paginator': paginator,
            'page': paginator.page,  
            'page_count': paginator.page.paginator.num_pages   # Pass the total number of cars for pagination info
        })

    




# Updated recommend_similar_cars function
def recommend_similar_cars(car_model):
    # Get all cars from the database
    cars = Car.objects.all()

    # Check if the QuerySet is empty
    if not cars.exists():
        print("No cars found in the database.")
        return pd.DataFrame()  # Return an empty DataFrame if no cars found

    # Create a DataFrame from the QuerySet
    df_cars = pd.DataFrame(list(cars.values()))

    # Ensure the DataFrame is not empty
    if df_cars.empty:
        print("No cars found in the database.")
        return pd.DataFrame()  # Return an empty DataFrame if no cars found


    # Check if the car model exists in the DataFrame
    if car_model not in df_cars['car_model'].values:
        print(f"Car model '{car_model}' not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame if car model is not found

    # Get the index of the specified car model
    car_index = df_cars[df_cars['car_model'] == car_model].index[0]

    # Transform the features of the selected car using the preprocessor
    car_features = car_pipeline.named_steps['preprocessor'].transform(df_cars.iloc[[car_index]])

    # Get similar cars based on the selected car's features
    distances, indices = car_pipeline.named_steps['knn'].kneighbors(car_features)


    # Ensure all indices are valid
    valid_indices = [i for i in indices[0] if i < df_cars.shape[0]]

    # Check if any valid indices were found
    if not valid_indices:
        print(f"No similar cars found for car model '{car_model}'.")
        return pd.DataFrame()  # Return an empty DataFrame if no similar cars are found

    # Get the similar cars and reset the index for display
    similar_cars = df_cars.iloc[valid_indices].reset_index(drop=True)

    return similar_cars


# Updated car_detail view
@api_view(['GET'])
def car_detail(request, car_id):
        try:
           access_token = get_valid_access_token_user(request)
           response = Response({"message": "Your view's response data"})

           if access_token != request.COOKIES.get('access'):
                  response.set_cookie(
                'access', access_token,
                httponly=True,
                secure=True,
                samesite='Strict',
                max_age=['ACCESS_TOKEN_LIFETIME'].seconds  
            )
        except AuthenticationFailed:
                return Response({'error': "Unauthorized access. Token is invalid or expired."}, status=401)
        except Car.DoesNotExist:
           return Response({'error': "Car not found"}, status=404)

        # Retrieve the specific car by ID
        car = Car.objects.get(id=car_id)

        # Serialize the car object
        serializer = CarSerializer(car)
        car_model = car.car_model

        # Get recommendations based on the car model
        recommendations = recommend_similar_cars(car_model)

        # Check if the recommendations DataFrame is not empty
        if recommendations is not None and not recommendations.empty:
            # Convert to dictionary and remove the current car from recommendations
            recommendations_dict = recommendations.to_dict(orient='records')
            recommendations = [rec for rec in recommendations_dict if rec['id'] != car_id]  # Remove the current car
        else:
            recommendations = None  # Handle empty case as per your needs

        # Render the car details in the HTML template
        return render(request, 'car_detail.html', {
            'car': serializer.data,
            'recommendations': recommendations  # Pass recommendations to the template
        })








