# Generated by Django 2.0 on 2018-03-16 14:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('songs', '0002_auto_20180316_1422'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lyric',
            name='lyrics',
            field=models.CharField(max_length=50000),
        ),
    ]