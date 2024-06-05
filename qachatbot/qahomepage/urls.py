from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Map root URL to home view
]
