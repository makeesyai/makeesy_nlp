"""
Download Data
https://www.kaggle.com/team-ai/spam-text-message-classification (spam)
https://www.kaggle.com/c/nlp-getting-started/data (tweet)
Install pandas (data loading)

1. Data loading
2. Indexing (string-labels to ids)
3. Split train, test (85/15)

4. Data Iterator

5. Train the model on Spam detection task
6. Add Tweeter data (train and evaluate)
"""
import time

from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
import torch
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from text_classification.model_utils import set_seed

# Set seed for results reproduction
set_seed(123)


class Batcher(object):
    def __init__(self, x, y, batch_size=32):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_samples = x.shape[0]
        # self.indices = torch.arange(self.num_samples)
        # self.rnd = np.random.RandomState()
        self.indices = torch.randperm(self.num_samples)
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer + self.batch_size > self.num_samples:
            self.indices = torch.randperm(self.num_samples)
            # self.rnd.shuffle(self.indices)
            self.pointer = 0
            raise StopIteration
        else:
            batch = self.indices[self.pointer:self.pointer + self.batch_size]
            self.pointer += self.batch_size
            return self.x[batch], self.y[batch]


class Classifier(nn.Module):
    def __init__(self, embedding_size, num_labels, dropout=0.7):
        super(Classifier, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Linear(embedding_size, num_labels)

    def forward(self, x):
        tensor = self.dropout(x)
        logits = self.ff(tensor)
        return logits, F.softmax(logits, dim=-1)


# data = pd.read_csv("../data/nlp-getting-started/train.csv", dtype=object)
# data = pd.read_csv("../data/spam/SPAM text message 20170820 - Data.csv", dtype=object)
data = pd.read_csv("../data/disaster-response-en-fr/disaster_response_messages_training.csv",
                   dtype={'message': str, 'original': str, 'request': int})

data.rename(columns={
    'Message': 'text',
    'Category': 'target',
    'message': 'text',
    'original': 'text_fr',
    'related': 'target'
},
    inplace=True)

data = data[['text', 'text_fr', 'target']]
data.dropna(inplace=True)

# Convert String Labels to integer labels
le = LabelEncoder()
le.fit(data.target)
data['labels'] = le.transform(data.target)

X_train, X_test, y_train, y_test = \
    train_test_split(data.text, data.labels, stratify=data.labels, test_size=0.15, random_state=123)

X_train_fr, X_test_fr, y_train_fr, y_test_fr = \
    train_test_split(data.text_fr, data.labels, stratify=data.labels, test_size=0.15, random_state=123)

sentences = X_train.tolist()
labels = torch.tensor(y_train.tolist())

# encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
encoder = SentenceTransformer('quora-distilbert-multilingual')
start_time = time.time()
sentence_embeddings = encoder.encode(sentences, convert_to_tensor=True)
print(f'The encoding time:{time.time() - start_time}')

use_cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if use_cuda else 'cpu'

num_sentence, embedding_dim = sentence_embeddings.shape
batch_size = 16
training_batcher = Batcher(sentence_embeddings, labels, batch_size=batch_size)

num_labels = np.unique(labels).shape[0]
classifier = Classifier(embedding_dim, num_labels, dropout=0.01)

classifier.to(device)

optimizer = optim.Adam(classifier.parameters())
loss_fn = nn.CrossEntropyLoss()

test_sentences = X_test.tolist()
test_sentences_fr = X_test_fr.tolist()
test_labels = y_test.tolist()
test_labels_fr = y_test_fr.tolist()

test_sentence_embeddings = encoder.encode(test_sentences_fr,
                                          convert_to_tensor=True)
test_sentence_embeddings = test_sentence_embeddings.to(device)

# Run prediction before training
with torch.no_grad():
    model_outputs, prob = classifier(test_sentence_embeddings)
    predicted_labels = torch.argmax(prob, dim=-1)
    accuracy = classification_report(predicted_labels.data.tolist(), test_labels_fr)
    print(f'Accuracy before training:')
    print(accuracy)

total_loss = 0
for e in range(30):
    updates = 0
    for index, batch in enumerate(training_batcher):
        updates += 1
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        model_outputs, prob = classifier(x)
        loss = loss_fn(model_outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch: {e}, Average loss:{total_loss/updates}')
    total_loss = 0
# if index and index % 100 == 0:


with torch.no_grad():
    model_outputs, prob = classifier(test_sentence_embeddings)
    predicted_labels = torch.argmax(prob, dim=-1)
    accuracy = classification_report(predicted_labels.data.tolist(), test_labels_fr)
    print(f'Accuracy after training:')
    print(accuracy)
    print(confusion_matrix(predicted_labels.data.tolist(), test_labels_fr))
