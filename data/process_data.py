# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories by dropping the last two characters
    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace the original 'category' column in df with categories df
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df,categories,left_index=True,right_index=True)

    # drop duplicates from the 'message' column
    df.drop_duplicates(subset='message',inplace=True)

    return df

def save_data(df, database_filename, table_name):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table_name, engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        table_name = 'merged'

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}\n    TABLE: {}'.format(database_filepath, table_name))
        save_data(df, database_filepath, table_name)

        print('Cleaned data saved to database!')

    elif len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath, table_name = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}\n    TABLE: {}'.format(database_filepath, table_name))
        save_data(df, database_filepath, table_name)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. Optional: provide the table name to '\
              'save to as well.\n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db (optional)merged')


if __name__ == '__main__':
    main()
