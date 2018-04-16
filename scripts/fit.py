import pandas as pd



# A function for fitting to a TDIDF Vectorizer
def fit(vect):
	lyric_dataset = pd.read_csv('./res/lyric_dataset.csv', sep=',')
	vect.fit(lyric_dataset['lyrics'])
	return vect