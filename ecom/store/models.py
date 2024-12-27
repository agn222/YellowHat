from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class customer(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE, null=True,blank=True)
    name = models.CharField(max_length=200, null=True)
    email = models.CharField(max_length=200 , null=True)

    def __str__(self):
       return self.name


class Product(models.Model):
    name = models.CharField(max_length=200, null=True)
    price = models.FloatField()
    digital = models.BooleanField(default=True, null=True, blank=False)

    def __str__(self):
        return self.name
    

class order(models.Model):
    customer = models.ForeignKey(customer, on_delete=models.SET_NULL, blank=True,null=True)
    date_ordered = models.DateTimeField(auto_now_add=True)
    complete = models.BooleanField(dafault=False, null=True, blank= False)

    def __str__(self):
        return str(self.id)

         

