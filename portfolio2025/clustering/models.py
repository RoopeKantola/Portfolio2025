from django.db import models

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=5000)
    url = models.CharField(max_length=500)

    def __str__(self):
        return self.name



