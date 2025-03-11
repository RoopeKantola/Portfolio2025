from django.db import models

# Create your models here.
class Project(models.Model):

    name = models.CharField(max_length=100)
    description = models.CharField(max_length=5000)
    image = models.ImageField()

    def __str__(self):
        return self.name


