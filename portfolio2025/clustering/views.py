from django.shortcuts import render
from main.models import Project
from .testfile import draw_graph


# Create your views here.
def home(response):
    projects = Project.objects.all()
    print(projects)

    context = {
        "projects": projects,
    }

    return render(response, "clustering/home.html", context=context)


def index(response, id):
    project = Project.objects.get(id=id)
    graph = draw_graph()
    print(project)

    context = {
        "project": project,
        "graph": graph,
    }

    return render(response, "clustering/index.html", context=context)
