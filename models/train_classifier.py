# import libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import string
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, recall_score
from sklearn.svm import LinearSVC

def load_data(database_filepath, table_name):
    '''
    Load dataset from database

    Inputs:
    database_filepath: string, location of database
    table_name: string, name of table in database

    Returns:
    X: pandas DataFrame, input parameters (messages)
    y: pandas DataFrame, output classifications (message categories)
    category_names: list, names of categories in y

    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name,engine)
    X = df[['message']]
    y = df.iloc[:,-36:]
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    '''
    Steps:
    1. Normalize (make lowercase and remove punctuation)
    2. Split into words
    3. Remove stop words
    4. Lemmatize

    Inputs
    text: str, text to tokenize

    Returns
    tokens: list, list of tokens (strings)
    '''
    # 1. Normalize (make lowercase and remove punctuation)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)

    # 2. Split into words
    words = word_tokenize(text)

    # 3. Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # 4. Lemmatize
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words]

    return tokens

def pos_tokenizer(text):
    '''
    Part of Speech tokenizer. Takes text, splits it into sentences,
    identifies the part of speech of each word in the sentence,
    extracts those parts of speech and returns them in a list.

    Input:
    text: string, raw text to tokenize - very important that no words are removed or lemmatized.

    Returns:
    pos_tags: list of the parts of speech of the input text
    '''
    pos_tags = []
    # tokenize by sentences
    sentence_list = sent_tokenize(text)

    for sentence in sentence_list:
        # tokenize each sentence into words, tag and extract part of speech
        pos_list = [i[1] for i in pos_tag(word_tokenize(sentence.lower()))]

        # remove punctuation tags (kept in sentence tokens to provide context to POS tagging)
        for element in pos_list:
            if element in string.punctuation:
                pos_list.remove(element)
        pos_tags += pos_list

    return pos_tags

def build_model():
    '''
    Builds a model pipeline and a grid search object

    Returns: model, grid search object refit with best parameters
    '''
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('pos_tfidf', TfidfVectorizer(tokenizer=pos_tokenizer))
    ])),

    ('mult_clf', MultiOutputClassifier(LinearSVC(loss='hinge', class_weight='balanced', max_iter=10000)))
    ])

    # specify parameters for grid search
    # # These were the initial grid search parameters - takes about 3 hrs to run
    # parameters = {
    # 'features__tfidf__max_features': [10000, None],
    # 'features__tfidf__ngram_range': [(1,1),(1,2)],
    # 'features__pos_tfidf__ngram_range': [(1,1),(1,2)],
    # 'mult_clf__estimator__loss': ['hinge', 'squared_hinge']
    # }

    # specifying a single parameter to demonstrate a quick grid search
    parameters = {
    'mult_clf__estimator__C': [0.1, 1.0, 2.0]
    }

    # Create a scorer to evaluate model performance on average recall score during grid search
    scorer = make_scorer(recall_score, average='macro')

    # create grid search object
    model = GridSearchCV(pipeline, parameters, cv=5, scoring=scorer)

    return model

def evaluate_model(model, X_test, y_test, category_names):
    '''
    Performs predictions on test data and displays the classification report
    for each category as well as the best parameters from the grid search

    Inputs:
    model: grid search object that has been fit to training data
    X_test: array of input features for test data
    y_test: array of output classifications for test data
    category_names: list, names of categories in y_test

    Returns:
    None
    '''

    y_preds = model.predict(X_test)

    for col in range(y_test.shape[1]):
        print(category_names[col],'\n',classification_report(y_test.to_numpy()[:,col], y_preds[:,col]))

    print('\nBest Parameters:', model.best_params_)


def save_model(model, model_filepath):
    '''
    Saves the trained model to a pickle file

    Inputs:
    model: trained model
    model_filepath: string, location of output pickle file

    Returns:
    None
    '''
    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
    file.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        table_name = 'merged'
        print('Loading data...\n    DATABASE: {}\n    TABLE: {}'.format(database_filepath,table_name))
        X, y, category_names = load_data(database_filepath, table_name)
        # Drop 'child_alone' category since it only has value 0
        y.drop(columns='child_alone',inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X['message'], y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    elif len(sys.argv) == 4:
        database_filepath, table_name, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}\n    TABLE: {}'.format(database_filepath,table_name))
        X, y, category_names = load_data(database_filepath, table_name)
        # Drop 'child_alone' category since it only has value 0
        y.drop(columns='child_alone',inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X['message'], y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. If the name of the '\
              'table in the disaster messages database is not "merged", please '\
              'provide the table name as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db table_name(optional) classifier.pkl')


if __name__ == '__main__':
    main()
