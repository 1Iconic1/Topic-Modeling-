import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
#load df
file = pd.read_csv('clean_sample.csv')
source = pd.read_csv('arxiv_train_sample.csv')

class Embeddings:

    def __init__(self, file_path ='Word2Vec-Embeddings/glove.6B.100d.txt' ):
        self.embeddings = {}
        with open(file_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings[word] = coefs

    def cosine_similarity(self, w1,w2):
        #if w1 in self.embeddings and w2 in self.embeddings:
        w1 = self.doc_embeddings(w1)
        w2 = self.doc_embeddings(w2)
        dot = np.sum(w1 * w2)

        #vector norm
        norm = np.sqrt(np.sum(w1**2)) *  np.sqrt(np.sum(w2**2))
        #print(dot,norm)
        similarity = dot / norm if norm != 0 else 0.0
        #else:
            #return 0

        return similarity

    def doc_embeddings(self,doc):

        #tokens = word_tokenize(doc)
        vectors = []
        for token in doc:
            if token in self.embeddings:
                vectors.append(self.embeddings[token])
        if vectors == [] :
            return np.zeros_like(self.embeddings['random'])
        doc_embedding = np.sum(vectors)# axis=0
        return doc_embedding
    #finds the most similar based on cosine similarity of each document
    def most_similar(self, doc, data):

        #doc1 = self.doc_embeddings(doc)
        tokens = word_tokenize(doc)
        similarities = [(self.cosine_similarity(tokens,vec),vec) for vec in data]
        similarities.sort(key=lambda x: x[0], reverse=True)

        return similarities[-1]




vec =  Embeddings()
#test cases on 3 article titles
test_articles= ['Extensive-Form Game Solving via Blackwell Approachability on Treeplexes','Towards Trust and Reputation as a Service in a Blockchain-based Decentralized Marketplace','Fair Artificial Currency Incentives in Repeated Weighted Congestion Games: Equity vs. Equality']
for title in test_articles:
    data = vec.most_similar(title.lower(), file['title'])
    #get non-tokenized article title
    '''
    filtered = file[file['doi']] == data
    if not filtered.empty:
        doi_value = filtered['doi'].iloc[0] 
        corresponding_title = source.loc[source['doi'] == doi_value, 'title'].iloc[0]
        data = corresponding_title
    '''
    print(data)
