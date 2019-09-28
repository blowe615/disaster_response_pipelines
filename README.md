# Disaster Response Pipeline Project

## Table of Contents
* [Installations](#installations)
* [Project Motivation](#project-motivation)
* [File Descriptions](#file-descriptions)
* [Instructions](#instructions)
* [Results](#results)
  - [Classifier](#classifier)
  - [Web App](#web-app)
* [Acknowledgements](#acknowledgements)

## Installations
All of the following packages were used for this project and can be found in the standard Anaconda distribution for Python 3.7:
* Jupyter notebook
* NumPy
* Pandas
* Scikit-learn
* NLTK
* Flask
* Plotly

## Project Motivation
As part of Udacity's [Data Scientist Nanodegree](https://www.udacity.com/school-of-data-science) program, I was tasked with creating an ETL (Extract Transform Load) pipeline followed by a machine learning pipeline to perform Natural Language Processing (NLP) on messages sent during disasters.

## File Descriptions
There are four directories in this repository:
#### data
* `disaster_messages.csv`: CSV file containing over 26,000 messages sent during natural disasters.  The dataset used to train and test the classifier.
* `disaster_categories.csv`: CSV file containing 36 category labels for the messages in `disaster_messages.csv`.
* `process_data.py`: Python script to run the ETL pipeline.  It takes 3 or 4 arguments: messages_filepath, categories_filepath, database_filepath, table_name (optional, default='merged').  See Step 1 of Instructions.
* `DisasterResponse.db`: SQLite database containing the merged, transformed and cleaned data that is ready for the machine learning pipeline

#### notebooks
* `ETL Pipeline Preparation Steps.ipynb`: Jupyter Notebook containing the workflow I used to create the functions in the `process_data.py` script.  Feel free to explore this to see my thought process and some outputs from the ETL process.
* `ML Pipeline Preparation.ipynb`: Jupyter Notebook containing the workflow I used to create the functions in `train_classifier.py` script.  This includes experimentation with different classifiers and hyperparameters that allowed me to arrive at my final model.

#### models
* `train_classifier.py`: Python script to run the ML pipeline that builds the model, trains it, tunes hyperparameters using GridSearch, displays the model performance results on the test data, and saves the best model to a pickle file.  It takes 2 or 3 arguments:
database_filepath, table_name (optional, default='merged'), model_filepath.  See Step 1 of Instructions.
* `model.pkl`: pickle file containing trained model.

#### app
* `run.py`: Python script to run the web app displaying visualizations of the dataset and allowing users to input a message and view its categories.
* `templates`: directory containing 2 html files:
  -  `master.html`: Main page of web app. Contains visualizations of messages dataset.
  -  `go.html`: Classification result page of web app.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`


2. Run the following command in the `app` directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results
### Classifier
Since there were 36 classification categories, a MultiOutputClassifier was used to train the dataset.  The plot below shows the recall score of the classifier on each of the categories.

![](https://github.com/blowe615/disaster_response_pipelines/blob/master/model_performance.png)

All of the recall scores range from 0.75 to 0.95.  Ideally, the scores would all be above 0.90, but the imbalance in most of the categories posed a major challenge in training the model.  The plot below shows the frequency of occurrence of each of the categories in the messages dataset.  

![](https://github.com/blowe615/disaster_response_pipelines/blob/master/category_freqs.png)

Of the 36 categories, only 7 occurred in more than 10% of messages.  This made it incredibly difficult to train the classifiers and resulted in a less than ideal performance.  There were three steps that I took to try to handle the imbalanced data.

1. Drop the 'child_alone' category.  None of the messages in the dataset were classified as 'child_alone' meaning that it would be impossible to train a classifier to identify that category.  I simply dropped that category from the training and testing datasets, so I actually trained a MultiOutput Classifier on 35 categories instead of 36.

2. Use a Linear Support Vector Classifier because it has the option to weight the classes in each category so that they are balanced.  For example, if one category has 100 samples with a classification of 0 and only 10 samples with a classification of 1, then the 10 samples are given 10 times the weight of the 100 samples to increase the emphasis on training the model to the samples where the classification is true.

3. Use recall as the scoring parameter in GridSearch.  Recall is the number of true positives divided by the sum of the number of true positives and false negatives.  A low recall score will have a large number of false negatives, meaning that a model must reduce the number of false negatives to improve its recall score.  Considering that these messages are sent during disasters, when people are in need of help, I believe it is better to prioritize reducing the number of false negatives rather than false positives (precision score).  In that case, you may have more messages that are incorrectly classified into some categories, but you also have fewer messages that are omitted from those categories when they should be included.  It is much better to have an aid worker receive unimportant messages than it is for them to miss messages to which they should have responded.

### Web App
The main page of the web app shows a couple plots about the data set used during this project.

At the top of the page is a text input box where the user can input a message and view the classification results.

![](https://github.com/blowe615/disaster_response_pipelines/blob/master/message_results_filtered.png)

Only the categories that apply to the message are displayed.  If you wish to view all of the categories, uncomment line 17 in `/app/templates/go.html`.

![](https://github.com/blowe615/disaster_response_pipelines/blob/master/message_results_unfiltered.png)

## Acknowledgements
This project was created by Udacity in partnership with [Figure Eight](https://www.figure-eight.com/), who provided the pre-labeled data set.  Thank you Figure Eight for collecting and sharing this data set and allowing me to work on a project with real world implications.  Stack Overflow posts and the documentation for each of the python packages were extremely helpful in completing this project.
