from django.contrib import admin
from .models import Record, Example, Patient, Physician

class ExampleAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')

#Register models here
admin.site.register(Record)
admin.site.register(Example, ExampleAdmin)
admin.site.register(Patient)
admin.site.register(Physician)