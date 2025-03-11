from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import Project

# Create your views here.
def home(response):
    projects = Project.objects.all()[:8]

    context = {
        "projects": projects,
    }
    print(projects)
    return render(response, "main/home.html", context=context)

