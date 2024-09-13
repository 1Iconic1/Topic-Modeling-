import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random

#load file  
file = pd.read_csv('clean_sample.csv')


def data_norm():
    data = [title for title in file['title']]
    return data
data = data_norm()

class LDA:
    def __init__(self, num_topics, num_iterations=1, alpha=1, beta=1):
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.topic_probs =[0 for i in range(self.num_topics)]

    def fit(self, data):
        self.vocabulary = list(set(data))
        self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocabulary)}
        num_documents = len(data)
        num_words = len(self.vocabulary)

        self.topic_assignments = {}

        self.doc_topic_counts = {doc_id: Counter() for doc_id in range(num_documents)}
        self.topic_word_counts = {z: Counter() for z in range(self.num_topics)}
        self.topic_counts = Counter()
        self.doc_lengths = [len(doc) for doc in data]

        # random topic assignment and gather counts
        for d in range(num_documents):
            for w in range(len(data[d])):
                self.topic_assignments[d, w] = np.random.randint(0, self.num_topics)
                topic = self.topic_assignments[d, w]
                self.doc_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][data[d][w]] += 1
                self.topic_counts[topic] += 1

        # Gibbs sampling
        for iteration in range(self.num_iterations):
            if iteration % 2 == 0:
                print("\n\nIteration:", iteration)
            for d in range(num_documents):
                for w in range(len(data[d])):
                    # Update counts
                    topic = self.topic_assignments[d,w]
                    self.doc_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][data[d][w]] -= 1
                    self.topic_counts[topic] -= 1

                    # calculate probability distribution
                    self.topic_probs[topic] = (self.doc_topic_counts[d][topic] + self.alpha) * (self.topic_word_counts.get(data[d][w],0) + self.beta) / (
                                          self.topic_counts[topic] + self.beta * len(self.vocabulary))


                    # assign a new topic
                    weights = [prb for prb in self.topic_probs]
                    new_topic = random.choices(range(self.num_topics), weights = weights)[0]

                    
                    self.topic_assignments[d, w] = new_topic

                    # Update counts with the new assignment
                    self.doc_topic_counts[d][new_topic] += 1
                    self.topic_word_counts[new_topic][data[d][w]] += 1
                    self.topic_counts[new_topic] += 1

    def get_results(self):
        topic_dist = [ [key,max(counts, key=counts.get)] for key,counts in self.doc_topic_counts.items()]
        self.doc_topic_counts = [ [self.idx2word[key],max(counts, key=counts.get)] for key,counts in self.doc_topic_counts.items() if key in self.idx2word.keys()]

        return self.doc_topic_counts, self.topic_word_counts.values(), self.topic_assignments, self.vocabulary,topic_dist, self.topic_counts

# Create LDA model and fit
lda_model = LDA(num_topics=2)

lda_model.fit(data)
print('finished fitting...')

# results
df_doc_topic, df_topic_word, topic_assignments, vocabulary,topic_dist,topic_counts = lda_model.get_results()

#print(df_doc_topic)
plt.figure(figsize=(10, 6))
plt.imshow(topic_dist, cmap='viridis', aspect='auto')
plt.colorbar(label='Topic Proportion')
plt.title('Document-Topic Matrix')
plt.xlabel('Topics')
plt.ylabel('Documents')
plt.show()


topic = list(topic_counts.keys())
titles = [val for val in list(topic_counts.values())]

    
print('creating topic assignments...')

plt.figure(figsize=(8, 6))
plt.bar(topic,titles)
plt.xlabel('Topic Assignment')
plt.ylabel('Documents')
plt.title('Topic Assignments Bar Plot')
plt.xticks(topic)
plt.show()
