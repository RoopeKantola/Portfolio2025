from django.shortcuts import render
from main.models import Project

# Create your views here.
def home(response):
    projects = Project.objects.all()
    print(projects)

    context = {
        "projects": projects,
    }
    return render(response, "swarm/home.html", context=context)
