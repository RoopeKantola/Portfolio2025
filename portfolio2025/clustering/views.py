from django.shortcuts import render
from main.models import Project
from .algorithms.kMeans import kMeans


# Create your views here.
def home(response):
    projects = Project.objects.filter(subject="clustering")
    print(projects)

    context = {
        "projects": projects,
    }

    return render(response, "clustering/home.html", context=context)


def index(response, id):
    project = Project.objects.get(id=id)

    context = {
        "project": project,
    }

    return render(response, "clustering/index.html", context=context)
