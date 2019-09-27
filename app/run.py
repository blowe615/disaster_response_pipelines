import json
import plotly
import pandas as pd
import string
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('merged', engine)
df.drop(columns='child_alone', inplace=True)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_freqs = df.iloc[:,4:].mean(axis=0).sort_values(ascending=False)
    category_names = category_freqs.index
    cats_per_message = df.iloc[:,4:].sum(axis=1).tolist()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_freqs
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                }
            }
        },
        {
            'data': [
                {
                    'x': cats_per_message,
                    'type': 'histogram'
                },
            ],

            'layout': {
                'title': 'Number of Categories per Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories per Message",
                    'automargin': True
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
