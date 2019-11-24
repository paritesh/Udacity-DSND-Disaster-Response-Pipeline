import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='left', left_on='id', right_on='id').drop(labels=["id"], axis=1)
    return df
def clean_data(df):
   
    categories = df.categories.str.split(";", expand=True)
    row = categories.loc[0]
    row[:5]
    category_colnames = []
    for i in row:
        category_colnames.append(i.split("-")[0])

    print(category_colnames)
    categories.columns = category_colnames
    for col in categories.columns: 
        categories[col] =categories[col].str.strip().str[-1]
    categories = categories.astype(int)
    df=df.drop(labels=["categories"], axis=1)
    df = df.join(categories)
    df.duplicated().sum()
    df = df.drop_duplicates()
    df.duplicated().sum()
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()