from django.urls import path, include
from . import views

urlpatterns = [
    path('demo/', views.swarm_demo, name="demo"),
]

