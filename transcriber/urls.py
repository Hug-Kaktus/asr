from django.urls import path
from . import views

app_name = 'transcriber'

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('result/<int:pk>/', views.result_view, name='result'),
]
