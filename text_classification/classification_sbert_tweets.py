"""
Download Data
https://www.kaggle.com/c/nlp-getting-started/data
Install pandas (data loading)
"""

import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from torch import nn, optim
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Batcher(object):
    def __init__(self, x, y, batch_size=32, seed=123):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_samples = x.shape[0]
        self.indices = np.arange(self.num_samples)
        self.rnd = np.random.RandomState(seed=seed)
        self.rnd.shuffle(self.indices)
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer + self.batch_size > self.num_samples:
            self.rnd.shuffle(self.indices)
            self.pointer = 0
            raise StopIteration
        else:
            batch = self.indices[self.pointer:self.pointer + self.batch_size]
            self.pointer += self.batch_size
            return torch.tensor(self.x[batch]), torch.tensor(self.y[batch], dtype=torch.long)


class Classifier(nn.Module):
    def __init__(self, embedding_size, num_labels, dropout=0.7):
        super(Classifier, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Linear(embedding_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        tensor = self.ff(x)
        return F.softmax(tensor, dim=-1)


data = pd.read_csv("../data/nlp-getting-started/train.csv")
X_train, X_test, y_train, y_test = \
    train_test_split(data.text, data.target, stratify=data.target, test_size=0.25)

sentences = X_train.tolist()
labels = np.array(y_train.tolist())

# embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
start_time = time.time()
sentence_embeddings = embedder.encode(sentences)
print(f'The encoding time:{time.time() - start_time}')

use_cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if use_cuda else 'cpu'

num_sentence, embedding_dim = sentence_embeddings.shape
training_batcher = Batcher(sentence_embeddings, labels, batch_size=32)

num_labels = np.unique(labels).shape[0]
classifier = Classifier(embedding_dim, num_labels, dropout=0.7)

classifier.to(device)

optimizer = optim.Adam(classifier.parameters())
loss_fn = nn.CrossEntropyLoss()

# with torch.no_grad():
#     predict = classifier(sentence_embeddings)
#     print(torch.argmax(predict, dim=-1))
test_sentences = X_test.tolist()
test_labels = y_test.tolist()

test_sentence_embeddings = embedder.encode(test_sentences,
                                           convert_to_tensor=True)
test_sentence_embeddings = test_sentence_embeddings.to(device)

with torch.no_grad():
    predict = classifier(test_sentence_embeddings)
    predicted_labels = torch.argmax(predict, dim=-1)
    accuracy = accuracy_score(predicted_labels.data.tolist(), test_labels) * 100
    print(accuracy)

total_loss = 0
for e in range(30):
    for index, batch in enumerate(training_batcher):
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        predict = classifier(x)
        loss = loss_fn(predict, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if index % 100 == 0:
            print(f'Epoch{e}, Average loss:{total_loss/100}')
            total_loss = 0

with torch.no_grad():
    predict = classifier(test_sentence_embeddings)
    predicted_labels = torch.argmax(predict, dim=-1)
    accuracy = accuracy_score(predicted_labels.data.tolist(), test_labels) * 100
    print(accuracy)
