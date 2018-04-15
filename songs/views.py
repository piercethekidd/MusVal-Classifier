from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from songs.models import Song


# Create your views here.

# Index
def index(request):
	wat = "HELLO! WORLD"
	context = {'wat': wat}
	return render(request, 'songs/index.html', context)


def display(request):
	song_list = Song.objects.order_by('id')[:1100]
	context = {'song_list': song_list,}
	return render(request, 'songs/display.html', context)

# Results
def results(request):
	return render(request, 'songs/results.html', {})

# Statistics
def statistics(request):
	return render(request, 'songs/statistics.html', {})

# Methodology
def methodology(request):
	return render(request, 'songs/methodology.html', {})

# Search
def search(request):
	return render(request, 'songs/search.html', {})

# Classify
def classify(request):
	return render(request, 'songs/classify.html', {})