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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


def run():

	# Read data from csv
	lyric_dataset = pd.read_csv('./res/lyric_dataset.csv', sep=',')
	feature_dataset = pd.read_csv('./res/feature_dataset.csv', sep=',')

	print('Lyrics and features list opened.')

	# Sort by song_id
	lyric_dataset = lyric_dataset.sort_values(by='song')
	feature_dataset = feature_dataset.sort_values(by='song')


	# Apply thresholding; Change label to either energy or valence
	feature_dataset.loc[feature_dataset['valence'] >= 0.5, "valence"] = 1
	feature_dataset.loc[feature_dataset['valence'] < 0.5, "valence"] = 0

	# Separate features and labels from dataset
	data_x = lyric_dataset['lyrics']
	data_y = feature_dataset['valence']

	
	# Initialize Count Vectorizer and Multinomial Naive Bayes as estimators for pipelining
	estimators = [('vectorizer', CountVectorizer(min_df=1, stop_words='english', lowercase=True, analyzer=Stem(), ngram_range=(1,2))),
	('clf', MultinomialNB())]
	pipe = Pipeline(estimators)
	print('Multinomial Naive Bayes initialized.')
	print('Performing K-Fold Cross Validation where K = 10')
	

	# Use K-Fold Cross Validation while using the pipeline as estimator
	scores = cross_val_score(pipe, data_x, data_y, cv=KFold(n_splits=10, shuffle=True))
	print("Scores: " + str(scores))
	print("Mean Score: %.4f" % scores.mean())

	# Model persistence; Save current model for future use
	joblib.dump(pipe, './res/mnb.pkl')


# Define a stemming callable to provide stemming for Lyrics Vectorizer
class Stem(object):

	def __init__(self):
		self.stemmer = SnowballStemmer('english')
		self.analyzer = CountVectorizer(min_df=1, stop_words='english', lowercase=True).build_analyzer()

	def __call__(self, doc):
		return [self.stemmer.stem(t) for t in self.analyzer(doc)]

