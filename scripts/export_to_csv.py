from songs.resources import *
import csv

def run():
	song_resource = SongResource()
	dataset = song_resource.export()
	text_file = open("./resources/dataset.csv", "w")
	text_file.write(dataset.csv)
	text_file.close()

	dataset = SongFeatureResource().export()
	text_file = open("./resources/feature_dataset.csv", "w")
	text_file.write(dataset.csv)
	text_file.close()

	dataset = LyricResource().export()
	text_file = open("./resources/lyric_dataset.csv", "w")
	text_file.write(dataset.csv)
	text_file.close()