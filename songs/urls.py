from django.urls import path

from . import views

app_name = 'songs'

urlpatterns = [
	path('', views.index, name='index'),
	path('display/', views.display, name="display"),
]