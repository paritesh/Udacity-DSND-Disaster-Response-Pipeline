# import libraries
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from time import time
def load_data(database_filepath):
    nltk.download('stopwords')
    print("Creating connection to database...")
    engine = create_engine('sqlite:///'+database_filepath)
    con = engine.connect()
    engine.table_names()
    df = pd.read_sql_table("messages", con)
    df.head()
    X = df["message"]
    Y = df.drop(labels=["message", "original", "genre"], axis=1)
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    # Normalise by setting to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Create tokens 
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatise words
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return clean_tokens


def build_model():
    forest = RandomForestClassifier(random_state=42)

    forest_pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(forest, n_jobs=-1))])
    #forest_pipeline.fit(X_train, Y_train)
    return forest_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    forest_Y_Pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print("Column {}: {}".format(i, col))
    
        y_true = list(Y_test.values[:, i])
        y_pred = list(forest_Y_Pred[:, i])
        target_names = ['is_{}'.format(col), 'is_not_{}'.format(col)]
        print(classification_report(y_true, y_pred, target_names=target_names))

    return
def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    return


def main():
    if len(sys.argv) == 3:
        print('x')
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath)
        print(model_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(database_filepath)
        print(model_filepath)
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