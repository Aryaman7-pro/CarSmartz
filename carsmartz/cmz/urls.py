from django.urls import path
from .views import index, predict

urlpatterns = [
    path('api/', index, name='index'),
    path('predict/', predict, name='predict'),
]