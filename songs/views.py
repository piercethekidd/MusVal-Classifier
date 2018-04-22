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
import pandas as pd
import scipy as sp
import numpy as np



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


# Classify Index
def classify_index(request):
	return render(request, 'songs/classify_index.html', {})

# Classify
def classify(request):

	vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem(), ngram_range=(1,2))
	vectorizer = fit(vectorizer)
	
	classifier = request.POST.get('classifier')
	lyrics = request.POST.get('lyrics')
	if classifier == "Multinomial Naive Bayes":
		clf = joblib.load('./res/mnb.pkl')
	else:
		clf = joblib.load('./res/svm_lyrics.pkl')
	
	lyrics_list = []
	lyrics_list.append(lyrics)
	x = vectorizer.transform(lyrics_list)
	valence = clf.predict(x)

	context = {
		'lyrics' : lyrics,
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
	track_album = track['album']['name']
	track_date = track['album']['release_date']

	track_id = track['id']


	api = genius.Genius('EvIwS8Hujru0G5Oxr8sulv9z5YLaml5gVIR9JlGGDVjomh-9LOmwSJBbyQzOqbZ3')
	song = api.search_song(title, artist)
	data = {
		'song':song.lyrics,
		'id' : track_id,
		'title': track_name,
		'artist': track_artist,
		'url': image_url,
		'album': track_album,
		'date': track_date
	}

	return JsonResponse(data)

def svm(request):
	# Setup auth tokens for GENIUS and SPOTIFY API
	scope = 'user-library-read'
	username = '12183890197'
	token = util.prompt_for_user_token(username, scope, client_id="b226f2bec10e4127a60ec75e26562562", 
		client_secret="4a5041b096554a9e896ffbf83a214008", redirect_uri="https://example.com/callback/")
	spotify = spotipy.Spotify(auth=token)

	track_id = request.POST.get('id')
	lyrics = request.POST.get('lyrics')

	features = spotify.audio_features(track_id)
	acousticness = features[0]['acousticness']
	danceability = features[0]['danceability']
	energy = features[0]['energy']
	instrumentalness = features[0]['instrumentalness']
	loudness = features[0]['loudness']
	tempo = features[0]['tempo']
	valence = features[0]['valence']

	track = spotify.track(track_id)
	artist = track['artists'][0]['name']
	title = track['name']
	url = track['album']['images'][0]['url']
	
	dic = {
		'acousticness': [acousticness],
		'danceability': [danceability],
		'energy': [energy],
		'instrumentalness': [instrumentalness],
		'loudness': [loudness],
		'tempo': [tempo],
		'valence': [valence],
	}

	df = pd.DataFrame(data=dic)
	vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem(), ngram_range=(1,2))
	vectorizer = fit(vectorizer)
	lyrics_list = []
	lyrics_list.append(lyrics)
	x = sp.sparse.hstack((vectorizer.transform(lyrics_list), df[['acousticness','danceability',
		'energy', 'instrumentalness', 'loudness','tempo']].values), format='csr')
	x = x.todense()

	clf = joblib.load('./res/svm_with_audio_features_pipe.pkl')
	y = clf.predict(x)
	print(y)

	context = {
		'artist': artist,
		'title': title,
		'url': url,
		'lyrics': lyrics,
		'acousticness': acousticness,
		'danceability': danceability,
		'energy': energy,
		'instrumentalness': instrumentalness,
		'loudness': loudness,
		'tempo': tempo,
		'valence': valence,
		'predicted': y,
	}
	return render(request, 'songs/svm.html', context)


# Define a stemming callable to provide stemming for Lyrics Vectorizer
class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]	