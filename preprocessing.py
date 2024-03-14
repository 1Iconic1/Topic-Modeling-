import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#data = json.loads('arxiv-metadata-oai-snapshot.json')

#seed = 1
#sampled = df.sample(frac=0.05, random_state=seed)
#sampled.to_csv('arxiv_train_sample.csv', index=False)

df = pd.read_csv('arxiv_train_sample.csv')
drop = ['submitter','authors','comments','report-no','journal-ref','license', 'versions','update_date']
df = df.drop(columns = drop)

def clean_tokenize(text):
    # tokenize text and remove stopwords
    text = text.lower().strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return tokens

df['abstract'] = df['abstract'].apply(clean_tokenize)
df['title'] = df['title'].apply(clean_tokenize)

df.to_csv('clean_sample.csv', index=False)


    




    
        
            


