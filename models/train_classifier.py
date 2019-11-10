import sys

import numpy as np
import pandas as pd
import string


from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from sqlalchemy import create_engine


def load_data(database_filepath):
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DisasterResponse.db",engine)
    X = df['message'] 
    Y = df.iloc[:, 4:]
    categories = Y.columns
    return X, Y, categories


def tokenize(text):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    
    tokens = WhitespaceTokenizer().tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    
    final_tokens = []
    
    for token in tokens:
        token = token.lower().translate(table)
        token = lemmatizer.lemmatize(token)
        
        if token != "":
            final_tokens.append(token)
            
    return final_tokens


def build_model():
    pipeline = Pipeline([
    ("vect", CountVectorizer(tokenizer=tokenize)),
    ("tfidf", TfidfTransformer()),
    ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        "clf__estimator__n_estimators": [15, 20, 35, 50, 100],
        "clf__estimator__max_depth": [None, 50, 100],
        "clf__estimator__max_features": ["sqrt", "auto"]
    }

    cv = GridSearchCV(pipeline, parameters)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), target_names=category_names))


def save_model(model, model_filepath):
    import pickle
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()