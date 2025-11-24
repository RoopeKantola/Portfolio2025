import numpy as np
from django.shortcuts import render
from main.models import Project

# Create your views here.
def home(response):
    projects = Project.objects.filter(subject="tsp")
    print(projects)

    context = {
        "projects": projects,
    }
    return render(response, "tsp/home.html", context=context)


def index(response, id):
    project = Project.objects.get(id=id)

    print(project)

    context = {
        "project": project,
    }

    return render(response, "tsp/index.html", context=context)
