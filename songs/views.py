from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from songs.models import Song
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from scripts.fit import fit


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

# Define a stemming callable to provide stemming for Lyrics Vectorizer
class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]	