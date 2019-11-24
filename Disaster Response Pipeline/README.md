# Disaster Response Pipeline Project
A project that uses data engineering skills to analyze disaster data from to build a model for an API that classifies disaster messages. 

There are three components for this project in two stages.
The components are:
ETL Pipeline
ML Pipeline
Flask Web App

### Instructions:
Stage 1. Through Jupyter notebooks
This stage includes working with two jupyter notebook files:

ETL Pipeline Preparation.ipynb which is used to read, clean, and reload the data both in normal way and through pipelines.
ML Pipeline Preparation.ipynb which is used to construct a ML algorithm to predict the keywords from messages. The operation also includes both in normal way and through pipelines.
Stage 2. Updating the web app files
In tHis stage, we moved the final code from the jupyter notebooks to the following .py files:

Stage2\data\process_data.py for reading, cleaning, and reloading the data to SQL db.
The original data is in disaster_categories.csv and disaster_messages.csv files, and the output file is DisasterResponse.db

Stage2\models\train_classifier.py
The code reads the db file then exports the trained model to classifier.pkl file.

Stage2\app\run.py which run the web app.


