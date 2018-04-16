from django.urls import path

from . import views

app_name = 'songs'

urlpatterns = [
	path('', views.index, name='index'),
	path('display/', views.display, name="display"),
	path('results/', views.results, name="results"),
	path('statistics/', views.statistics, name="statistics"),
	path('methodology/', views.methodology, name="methodology"),
	path('classify/', views.classify_index, name="classify_index"),
	path('classify/results', views.classify, name="classify"),
	path('classify/search', views.search, name="search"),
	path('classify/search/ajax', views.ajax_search, name="ajax_search"),
	path('classify/svm-with-audio-features', views.svm, name="svm"),


]