from django.db import models

class Record(models.Model):
    firstName = models.CharField(max_length=50)
    lastName =  models.CharField(max_length=50)
    email =  models.CharField(max_length=100)
    dob =  models.CharField(max_length=50)
	
    def __str__(self):
       return(f"{self.first_name} {self.last_name}")

	
class Example(models.Model):
	name = models.CharField(max_length=120)
	description = models.TextField()

	def __str__(self):
		return self.description


class Physician(models.Model):
	firstName = models.CharField(max_length=50)
	lastName = models.CharField(max_length=50)
	email = models.CharField(max_length=100)
	special = models.CharField(max_length=50)


class Patient1(models.Model):
	firstName = models.CharField(max_length=50)
	lastName = models.CharField(max_length=50)
	email = models.CharField(max_length=100)
	gender = models.CharField(max_length=50)
	dob = models.DateField()

class Patient(models.Model):
	firstName = models.CharField(max_length=50)
	lastName = models.CharField(max_length=50)
	email = models.CharField(max_length=100)
	gender = models.CharField(max_length=50)
	dob = models.DateField()
	mri_image = models.FileField(upload_to='mri_images/')
	diagnosis = models.TextField()
	affected_areas = models.CharField(max_length=100)
	treatment_options = models.TextField()
	patient_id = models.AutoField(primary_key=True)

	def __str__(self):
		return f"{self.first_name} {self.last_name}"
