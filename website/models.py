from django.db import models


class Record(models.Model):
    firstName = models.CharField(max_length=50)
    lastName =  models.CharField(max_length=50)
    email =  models.CharField(max_length=100)
    dob =  models.CharField(max_length=50)


    def __str__(self):
        return(f"{self.firstName} {self.lastName}")