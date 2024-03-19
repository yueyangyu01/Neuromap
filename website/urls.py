from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from . import views
from django.urls import path
from .views import get_csrf_token

router = routers.DefaultRouter()
router.register(r'examples', views.ExampleView, 'example')
router.register(r'patient', views.PatientView, 'patient')
router.register(r'patients', views.PatientViewSet)
router.register(r'physician', views.PhysicianView, 'physician')

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    #path('login/', views.login_user, name='login'),
    path('logout/', views.logout_user, name='logout'),
    path('register/', views.register_user, name='register'),
    path('create/', views.create_user, name='create'),
    path('record/<int:pk>', views.customer_record, name='record'),
    path('delete_record/<int:pk>', views.delete_record, name='delete_record'),
    path('add_record/', views.add_record, name='add_record'),
    path('update_record/<int:pk>', views.update_record, name='update_record'),
    path('api/', include(router.urls)),
    path('api/create_patient/', views.create_patient, name='create_patient'),
    path('upload/', views.upload_file, name='upload_file'),
    path('create_physician/', views.create_physician, name='create_physician'),
    path('update_physician/<int:physician_id>/', views.update_physician, name='update_physician'),
    path('delete_physician/<int:physician_id>/', views.delete_physician, name='delete_physician'),
    path('list_patients/', views.list_patients, name='list_patients'),
    path('get_patient/<int:patient_id>/', views.get_patient, name='get_patient'),
    path('update_patient/<int:patient_id>/', views.update_patient, name='update_patient'),
    path('delete_patient/<int:patient_id>/', views.delete_patient, name='delete_patient'),
    path('api/', include(router.urls)),
    path('csrf-token/', get_csrf_token, name='get_csrf_token'),
]