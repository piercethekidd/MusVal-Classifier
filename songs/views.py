from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from songs.models import Song
from sklearn.externals import joblib
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from scripts.fit import fit
import lyricsgenius as genius
import spotipy
import spotipy.util as util


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

# Classify Index
def classify_index(request):
	return render(request, 'songs/classify_index.html', {})

# Classify
def classify(request):

	vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem(), ngram_range=(1,2))
	vectorizer = fit(vectorizer)
	
	classifier = request.POST.get('classifier')
	if classifier == "Multinomial Naive Bayes":
		clf = joblib.load('./res/mnb.pkl')
	else:
		clf = joblib.load('./res/svm_lyrics.pkl')
	
	lyrics_list = []
	lyrics_list.append(request.POST.get('lyrics'))
	x = vectorizer.transform(lyrics_list)
	valence = clf.predict(x)

	context = {
		'valence': valence,
		'classifier': classifier,
	}
	return render(request, 'songs/classify_result.html', context)

# Search
def search(request):
	return render(request, 'songs/search.html', {})

def ajax_search(request):
	artist = request.GET.get('artist', None)
	title = request.GET.get('title', None)

	# Setup auth tokens for GENIUS and SPOTIFY API
	scope = 'user-library-read'
	username = '12183890197'
	token = util.prompt_for_user_token(username, scope, client_id="b226f2bec10e4127a60ec75e26562562", 
		client_secret="4a5041b096554a9e896ffbf83a214008", redirect_uri="https://example.com/callback/")
	spotify = spotipy.Spotify(auth=token)

	query = artist + " " + title
	results = spotify.search(q=query, type='track')
	track = results['tracks']['items'][0]
	track_name = track['name']
	track_artist = track['artists'][0]['name']
	image_url = track['album']['images'][0]['url']

	api = genius.Genius('EvIwS8Hujru0G5Oxr8sulv9z5YLaml5gVIR9JlGGDVjomh-9LOmwSJBbyQzOqbZ3')
	song = api.search_song(title, artist)
	data = {
		'song':song.lyrics,
		'title': track_name,
		'artist': track_artist,
		'url': image_url,
	}

	return JsonResponse(data)


# Define a stemming callable to provide stemming for Lyrics Vectorizer
class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]	