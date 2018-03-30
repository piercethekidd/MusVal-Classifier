import datetime
from django.utils import timezone
from django.db import models

# Create your models here.
class Song(models.Model):
	title = models.CharField(max_length=200)
	artist = models.CharField(max_length=200)
	year = models.IntegerField(default=0)
	rank = models.IntegerField(default=0)

	def __str__ (self):
		return self.title



class SongFeature(models.Model):
	song = models.ForeignKey(Song, on_delete=models.CASCADE)
	acousticness = models.DecimalField(max_digits=5, decimal_places=4)
	danceability = models.DecimalField(max_digits=5, decimal_places=4) # ranges from -9.9999 to 9.9999
	energy = models.DecimalField(max_digits=5, decimal_places=4) # ranges from -9.9999 to 9.9999
	instrumentalness = models.DecimalField(max_digits=5, decimal_places=4) # ranges from -9.9999 to 9.9999
	loudness = models.DecimalField(max_digits=6, decimal_places=4) # ranges from -99.9999 to 99.9999
	tempo = models.DecimalField(max_digits=8, decimal_places=4) # ranges from -9999.9999 to 9999.9999
	valence = models.DecimalField(max_digits=5, decimal_places=4) # ranges from -9.9999 to 9.9999

	def __str__(self):
		return self.song.title + " Features"


class Lyric(models.Model):
	song = models.ForeignKey(Song, on_delete=models.CASCADE)
	lyrics = models.TextField("Lyrics Field")

	def __str__(self):
		return self.song.title + " Lyrics"