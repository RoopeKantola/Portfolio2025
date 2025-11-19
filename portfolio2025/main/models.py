from django.db import models

# Create your models here.
class Project(models.Model):

    name = models.CharField(max_length=100)
    layout_type = models.IntegerField(default=1)
    description_1 = models.TextField(max_length=10000)
    description_2 = models.TextField(max_length=10000, blank=True)
    description_3 = models.TextField(max_length=10000, blank=True)
    description_4 = models.TextField(max_length=10000, blank=True)

    image_1 = models.CharField(max_length=200, null=True, blank=False)
    subject = models.CharField(max_length=100, default="Non specific")
    animation_url_1 = models.CharField(max_length=200, null=True, blank=False)
    animation_url_2 = models.CharField(max_length=200, null=True, blank=True)
    animation_url_3 = models.CharField(max_length=200, null=True, blank=True)
    animation_url_4 = models.CharField(max_length=200, null=True, blank=True)

    def __str__(self):
        return self.name


