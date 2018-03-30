from import_export import resources
from .models import Song, SongFeature, Lyric

# To export models to csv we need to add resources.py
class SongResource(resources.ModelResource):
    class Meta:
        model = Song

class SongFeatureResource(resources.ModelResource):
    class Meta:
        model = SongFeature

class LyricResource(resources.ModelResource):
    class Meta:
        model = Lyric