from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('datasets/upload/', views.upload_dataset, name='upload_dataset'),
    path('datasets/<int:dataset_id>/configure/', views.configure_run, name='configure_run'),
    path('runs/<int:run_id>/', views.run_detail, name='run_detail'),
]

