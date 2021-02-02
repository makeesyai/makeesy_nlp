"""
1. 'distilbert-base-nli-mean-tokens'
2. 'quora-distilbert-multilingual' (multilingual)
Steps-
1. Create model (show each step with running the intermediate code)
2. Add Optimizer (Adam) and Loss function (CrossEntropy)
3. Add Testing with training and print evaluation
4. Add training Loop with logging loss
5. Add final evaluation code
"""
import time

from sentence_transformers import SentenceTransformer
from torch import nn, optim
import torch
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, embedding_size, n_labels, dropout=0.01):
        super(Classifier, self).__init__()
        self.embedding_size = embedding_size
        self.num_labels = n_labels
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Linear(embedding_size, n_labels)

    def forward(self, x):
        tensor = self.dropout(x)
        logits = self.ff(tensor)
        return logits, F.softmax(logits, dim=-1)


# Sentences we want sentence embeddings for
sentences = [
    'This framework generates embeddings for each input sentence',  # 0
    'Sentences are passed as a list of string.',  # 0
    'The quick brown fox jumps over the lazy dog.',  # 1
    'The lazy dog is also jumping.',  # 1
    'The fox and the dog are playing.'  # 1
]
labels = torch.tensor([0, 0, 1, 1, 1])

# encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
encoder = SentenceTransformer('quora-distilbert-multilingual')
start_time = time.time()

sentence_embeddings = encoder.encode(sentences, convert_to_tensor=True)
print(f'The encoding time:{time.time() - start_time}')

num_sentence, embedding_dim = sentence_embeddings.size()
num_labels = labels.unique().shape[0]
classifier = Classifier(embedding_dim, num_labels, dropout=0.01)

use_cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if use_cuda else 'cpu'

# Move model and data to device
classifier.to(device)
sentence_embeddings = sentence_embeddings.to(device)
labels = labels.to(device)

optimizer = optim.Adam(classifier.parameters())
loss_fn = nn.CrossEntropyLoss()

# with torch.no_grad():
#     predict = classifier(sentence_embeddings)
#     print(torch.argmax(predict, dim=-1))

for e in range(10):
    optimizer.zero_grad()
    model_outputs, prob = classifier(sentence_embeddings)
    loss = loss_fn(model_outputs, labels)
    print(loss.item())
    loss.backward()
    optimizer.step()

test_sentences = [
    'The sentence is used here has good embeddings.',
    'The boy is playing with the dog and jumping in joy.',
    'यहाँ वाक्य का इस्तेमाल किया गया है जिसमें अच्छी एम्बेडिंग है।',
    'डॉग्स के साथ खेलता लड़का और खुशी से उछलता।'
]
test_labels = [0, 1, 0, 1]

test_sentence_embeddings = encoder.encode(test_sentences, convert_to_tensor=True)
test_sentence_embeddings = test_sentence_embeddings.to(device)

with torch.no_grad():
    model_outputs, prob = classifier(test_sentence_embeddings)
    print(torch.argmax(prob, dim=-1))
