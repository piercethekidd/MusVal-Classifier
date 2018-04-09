import time as tm
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def run():
	lyric_dataset = pd.read_csv('./resources/lyric_dataset.csv', sep=',')
	feature_dataset = pd.read_csv('./resources/feature_dataset.csv', sep=',')

	print('Lyrics and features list opened.')

	# Sort by song_id
	lyric_dataset = lyric_dataset.sort_values(by='song')
	feature_dataset = feature_dataset.sort_values(by='song')


	# Change label to either energy or valence; apply thresholding
	feature_dataset.loc[feature_dataset['valence'] >= 0.5, "valence"] = 1
	feature_dataset.loc[feature_dataset['valence'] < 0.5, "valence"] = 0

	# Separate features and labels from dataset
	data_x = lyric_dataset['lyrics']
	data_y = feature_dataset['valence']
	#data_x = np.array(data_x)


	
	# Initialize Count Vectorizer and Multinomial Naive Bayes as estimators for pipelining
	estimators = [('vectorizer', CountVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem(), ngram_range=(2,2))),
	('clf', MultinomialNB())]
	pipe = Pipeline(estimators)
	print('Multinomial Naive Bayes initialized.')
	print('Performing K-Fold Cross Validation where K = 10')


	###### K-Fold Cross Validation
	######
	scores = cross_val_score(pipe, data_x, data_y, cv=KFold(n_splits=10, shuffle=True))
	print("Scores: " + str(scores))
	print("Mean Score: %.4f" % scores.mean())
	######
	######
	
class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]

