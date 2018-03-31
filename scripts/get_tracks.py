from songs.models import Song
from django.utils import timezone
import requests
from bs4 import BeautifulSoup

def run():
	print("Requests and bs4 imported\n\n")
	if(Song.objects.all().delete()):
		print("Deleted all files in table 'Song', 'Song Features', 'Song Lyrics'")



	for i in range(2006, 2017):
		r = requests.get("https://www.billboard.com/charts/year-end/" + str(i) + "/hot-100-songs")
		soup = BeautifulSoup(r.content, "html.parser")
		data = soup.find_all('div', {"class" : "ye-chart-item__primary-row"})

		print("Year: " + str(i) +"\n")
		for item in data:
			rank = item.contents[1].text
			title = item.contents[5].find_all("div", {"class" : "ye-chart-item__title"})[0].text
			artist = item.contents[5].find_all("div", {"class" : "ye-chart-item__artist"})[0].text

			print(rank.strip() + ". " + title.strip() + " - " + artist.strip())

			song = Song(title=title.strip(), artist=artist.strip(), year=i, rank=int(rank.strip()))
			song.save()


	for song in Song.objects.all():
		print(song)
		
	print("Total song count: " + str(Song.objects.count()))

	fp = open('resources/index_counter.txt', 'w')
	fp.write('0\n0')
