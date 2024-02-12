# Welcome to Neuromap's backend!

## Setting up Django

Install Django using `pip install django` (more info can be found in their [documentation](https://docs.djangoproject.com/en/5.0/faq/install/)).

To run migrations:
```
python manage.py makemigrations
python manage.py migrate
```
You can also specify the app by running: 
```
python manage.py makemigrations {app}
python manage.py migrate {app}
```

Run `python manage.py runserver` to start up the server, then navigate to `http://localhost:8000` in your browser. 

## Setting up MySQL

You can download the installer [here](https://dev.mysql.com/downloads/installer/). Accept the default configuration unless you have specific settings in mind.

## Connecting to the frontend

Code and instructions for Neuromap's frontend can be found in [this repo](https://github.com/zoeyz101/brain-tumor-segmentation-frontend).
