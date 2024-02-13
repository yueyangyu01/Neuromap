from django.contrib import admin
from .models import Record, Example

class ExampleAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')

#Register models here
admin.site.register(Record)
admin.site.register(Example, ExampleAdmin)