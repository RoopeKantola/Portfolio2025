from django.shortcuts import render

# Create your views here.
def swarm_demo(response):
    return render(response, "swarm/home.html")
