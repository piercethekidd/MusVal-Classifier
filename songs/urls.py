from django.urls import path

from . import views

app_name = 'songs'

urlpatterns = [
	path('', views.index, name='index'),
	path('display/', views.display, name="display"),
	path('results/', views.results, name="results"),
	path('statistics/', views.statistics, name="statistics"),
	path('methodology/', views.methodology, name="methodology"),
	path('search/', views.search, name="search"),

]