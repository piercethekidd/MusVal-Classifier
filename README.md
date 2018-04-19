# ProjectML

Special Problem Topic: Classifiying Musical Valence using Spotify Features as Audio Features

To start the project, install `pip` and `virtualenv`:
```
sudo apt-get install python3-pip
pip3 install virtualenv
```

Create a directory where you want your virtualenv and clone this repository to that directory and rename as 'src':
```
virtualenv -p python3 <directory_name>
cd <directory_name> && git clone https://github.com/slobaddik/ProjectML.git
mv ProjectML src
```
To activate or deactivate virtualenv, use:
```
source bin/activate // Enter on virtualenv root directory
deactivate // Can be entered anywhere on the virtualenv workspace
```

Install the latest version of PostgreSQL and setup postgres password:
```
sudo apt-get install postgresql postgresql-contrib
sudo su
su - postgres
psql
\password
```
Setup database and edit settings.py in `DATABASES`:
```
'NAME': '<database>',
'USER': 'postgres',
'PASSWORD': '<password>',
```

Install dependencies and apply migrations: 
```
pip install -r requirements.txt
python manage.py migrate
```

Use `python manage.py runscript <script_name>` for running scripts in the 'scripts' directory and `python manage.py runserver` to run the server.
