from django.db import models

# Create your models here.
class Filepath(models.Model):
    s3filepath = models.CharField(max_length=100)
    datakeyname = models.CharField(max_length=100)
    filestatus = models.CharField(max_length=100)
