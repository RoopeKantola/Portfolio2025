from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import Project

# Create your views here.
def base(response):
    projects = Project.objects.all()[:8]
    tsp_projects = Project.objects.filter(subject="tsp")
    clustering_projects = Project.objects.filter(subject="clustering")
    other_projects = Project.objects.filter(subject="other")

    context = {
        "projects": projects,
        "tsp_projects": tsp_projects,
        "clustering_projects": clustering_projects,
        "other_projects": other_projects,
    }
    print(projects)
    return render(response, "main/base.html", context=context)

