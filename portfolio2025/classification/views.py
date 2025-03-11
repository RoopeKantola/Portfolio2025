from django.shortcuts import render

# Create your views here.
def demo(response):
    return render(response, "classification/demo.html")

