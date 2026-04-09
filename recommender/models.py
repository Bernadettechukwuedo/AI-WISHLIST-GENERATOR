from django.db import models

# Create your models here.
class Prediction(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    category = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)