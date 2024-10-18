from django.urls import path
from .views import index, predict, depreciation_graph, user_registration, user_login, TokenRefreshView, logout_view, create_car, get_all_cars,car_detail, search_cars
from rest_framework_simplejwt.views import TokenRefreshView
urlpatterns = [
    path('api/', index, name='index'),
    path('predict/', predict, name='predict'),
    path('depreciation-graph/', depreciation_graph, name='depreciation_graph'),
    path('register/', user_registration, name='register'),
    path('login/', user_login, name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', logout_view, name='logout'),
    path('api/cars/sell/', create_car, name='car-create'),
    path('api/cars/buy/', get_all_cars, name='get-all-cars'),
    path('api/cars/<int:car_id>/', car_detail, name='car_detail'),
    path('api/cars/search/', search_cars, name='search_cars'),
]