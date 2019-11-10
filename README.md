# Disaster Response Pipeline Project

### Project Motivation
This main goal of this project is to apply data engineering skills to build a model for an application program interface (API) that 
classifies disaster messages based on data from Figure Eight. The model is used to categorize disaster events,so messages can be sent to an 
appropriate disaster relief agency. Anybody can get classification result by inputing a message via interface provided by the web app; 
additionally, the web app includes some visulization for contextual purposes.

### Dependencies
- Pandas 
- NumPy
- Scikit-Learn
- NLTK
- SQLalchemy
- Flask
- Plotly

### File Descriptions
```
. 
├── README.md
├── app
│   ├── run.py
│   └── templates 
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db
│   ├── YourDatabaseName.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── models
│   ├── classifier.pkl
│   └── train_classifier.py
└── notebook
    ├── ETL Pipeline Preparation.ipynb
    └── ML Pipeline Preparation.ipynb
```
#### Note:
- `YourDatabaseName.db` is not used and should be ignored.
- `process_data.py` contains the data pipeline, which is the Extract, Transform, and Load process. In this file,  the dataset is being 
read, cleaned, and stored it in a SQLite database.
- `train_classifier.py` is where the NLP and Machine Learning pipelines are implemented. The NLP pipeline involves the process of 
tokenizing, stemming, lemmatization, and feature extraction, while the Machine Learning pipeline includes train test splitting, 
training, evaluation, and optimization.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
