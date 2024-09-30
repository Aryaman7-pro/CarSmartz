from django.urls import path
from .views import index, predict, depreciation_graph, user_registration, user_login, TokenRefreshView
from rest_framework_simplejwt.views import TokenRefreshView
urlpatterns = [
    path('api/', index, name='index'),
    path('predict/', predict, name='predict'),
    path('depreciation-graph/', depreciation_graph, name='depreciation_graph'),
    path('register/', user_registration, name='register'),
    path('login/', user_login, name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]