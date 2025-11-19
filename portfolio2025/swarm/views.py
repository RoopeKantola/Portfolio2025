import numpy as np
from django.shortcuts import render
from main.models import Project
from .algorithms import PSO as pso

# Create your views here.
def home(response):
    projects = Project.objects.filter(subject="swarm")
    print(projects)

    context = {
        "projects": projects,
    }
    return render(response, "swarm/home.html", context=context)


def index(response, id):
    project = Project.objects.get(id=id)

    print(project)

    context = {
        "project": project,
    }

    return render(response, "swarm/index.html", context=context)
