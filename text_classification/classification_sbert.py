"""
Download Data
https://www.kaggle.com/team-ai/spam-text-message-classification (spam)
https://www.kaggle.com/c/nlp-getting-started/data (tweet)

Install pandas (data loading)

1. Data loading
2. Indexing (string-labels to ids)
3. Split train, test (85/15)
4. Add evaluation code with Precision/Recall/F1
5. Train the model for Spam detection task using Batch Gradiant Descent

# Load multilingual data for spam classification
6. Train the model on Spam detection task using Stochastic Gradiant Descent (optional)
7. Implement Batch Iterator
8. Train the model on Spam detection task using Mini Batch Gradiant Descent
9. Test Zero-shot accuracy
10. Add Tweeter data (train and evaluate)
"""
import time

import pandas
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim, tensor
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_labels, dropout):
        super(Classifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.dropout = dropout

        self.dp = nn.Dropout(self.dropout)
        self.ff = nn.Linear(self.embedding_dim, self.num_labels)

    def forward(self, input_embeddings):
        tensor = self.dp(input_embeddings)
        tensor = self.ff(tensor)
        return tensor, F.softmax(tensor, dim=-1)


data_df = pandas.read_csv("../data/spam/data-en-hi-de-fr.csv")
data_df.dropna(inplace=True)
data_df.drop_duplicates(inplace=True)
data_df.rename(columns={
    "Category": "labels",
    "Message": "text"
}, inplace=True)

le = LabelEncoder()
le.fit(data_df.labels)
data_df["labels"] = le.transform(data_df.labels)

train_x, test_x, train_y, test_y = \
    train_test_split(data_df.text, data_df.labels, stratify=data_df.labels, test_size=0.15)

sentences = train_x.tolist()
test_sentences = test_x.tolist()

labels = torch.tensor(train_y.tolist())
test_labels = torch.tensor(test_y.tolist())

# encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
encoder = SentenceTransformer('quora-distilbert-multilingual')
print('Encoding segments...')
start = time.time()
embedding = encoder.encode(sentences, convert_to_tensor=True)
test_sentences_embedding = encoder.encode(test_sentences, convert_to_tensor=True)
print(f"Encoding completed in {time.time() - start} seconds.")

num_samples, embeddings_dim = embedding.size()
n_labels = labels.unique().shape[0]

classifier = Classifier(embeddings_dim, n_labels, dropout=0.01)

optimizer = optim.Adam(classifier.parameters())
loss_fn = nn.CrossEntropyLoss()

for e in range(200):
    optimizer.zero_grad()
    model_output, prob = classifier(embedding)
    loss = loss_fn(model_output, labels)
    loss.backward()
    optimizer.step()
    print(loss.item())

with torch.no_grad():
    model_output, prob = classifier(test_sentences_embedding)
    predictions = torch.argmax(prob, dim=-1)
    results = classification_report(predictions, test_labels)
    print(results)
