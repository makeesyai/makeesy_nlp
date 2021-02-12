"""
Models:
1. 'distilbert-base-nli-mean-tokens'
2. 'quora-distilbert-multilingual' (multilingual)

Steps:
1. Create pytorch model for text classification (explain each step with running the intermediate code)
2. Add Testing with training data using random model

3. Add Optimizer (Adam) and Loss function (CrossEntropy)
4. Add training Loop with logging loss
5. Add final evaluation code
"""
import time

import torch
from sentence_transformers import SentenceTransformer
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


# Sentences we want sentence embeddings for
sentences = [
    'This framework generates embeddings for each input sentence',  # 0
    'Sentences are passed as a list of string.',  # 0
    'The quick brown fox jumps over the lazy dog.',  # 1
    'The lazy dog is also jumping.',  # 1
    'The fox and the dog are playing.'  # 1
]

labels = torch.tensor([0, 0, 1, 1, 1])

test_sentences = [
    'The sentence is used here has good embeddings.',  # 0
    'The boy is playing with the dog and jumping in joy.', # 1
    'यहाँ वाक्य का इस्तेमाल किया गया है जिसमें अच्छी एम्बेडिंग है।',  # 0 (MT Hindi Language)
    'डॉग्स के साथ खेलता लड़का और खुशी से उछलता।'  # 1 (MT Hindi Language)
]

test_labels = tensor([0, 1, 0, 1])

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

for e in range(10):
    optimizer.zero_grad()
    model_output, prob = classifier(embedding)
    loss = loss_fn(model_output, labels)
    loss.backward()
    optimizer.step()
    print(loss.item())


with torch.no_grad():
    model_output, prob = classifier(test_sentences_embedding)
    print(model_output, prob)
    print(f'True Labels: {test_labels}')
    print(f'Predicted Labels:{torch.argmax(prob, dim=-1)}')

