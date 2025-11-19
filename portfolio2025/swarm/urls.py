from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path("<int:id>", views.index, name="index"),
]

