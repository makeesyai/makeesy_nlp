from sentence_transformers import SentenceTransformer
from torch import nn, optim
import torch
from torch.nn import functional as F


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


# Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
labels = torch.tensor([0, 0, 1])

# embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

num_sentence, embedding_dim = sentence_embeddings.size()

use_cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if use_cuda else 'cpu'

classifier = Classifier(embedding_dim, 2, dropout=0.7)
if use_cuda:
    classifier.to(device)
    sentence_embeddings = sentence_embeddings.to(device)
    labels = labels.to(device)

optimizer = optim.Adam(classifier.parameters())
loss_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    predict = classifier(sentence_embeddings)
    print(torch.argmax(predict, dim=-1))

for e in range(10):
    optimizer.zero_grad()
    logits = classifier(sentence_embeddings)
    loss = loss_fn(logits, labels)
    print(loss)
    loss.backward()
    optimizer.step()

test_sentences = ['The sentence is used here has good embeddings.',
                  'The boy playing with the Dogs and jumping with joy.',
                  'यहाँ वाक्य का इस्तेमाल किया गया है जिसमें अच्छी एम्बेडिंग है।',
                  'डॉग्स के साथ खेलता लड़का और खुशी से उछलता।']
test_labels = [0, 1, 0, 1]

test_sentence_embeddings = embedder.encode(test_sentences, convert_to_tensor=True)

if use_cuda:
    test_sentence_embeddings = test_sentence_embeddings.to(device)

with torch.no_grad():
    predict = classifier(test_sentence_embeddings)
    print(torch.argmax(predict, dim=-1))
