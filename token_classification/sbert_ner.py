import math

import pandas
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from torch import nn
from torch import from_numpy
from torch.nn.functional import softmax
from transformers import XLNetTokenizer, T5Tokenizer, GPT2Tokenizer


def get_label2id_vocab(data):
    label2idx = {}
    idx = 0
    for line in data:
        for l in line:
            if l not in label2idx:
                label2idx[l] = idx
                idx += 1
    return label2idx


def get_label_ids(data, labels2ids):
    labels_ids = []
    for line in data:
        label_ids = []
        for l in line:
            label_ids.append(labels2ids.get(l, labels2ids.get('O')))
        labels_ids.append(label_ids)
    return labels_ids


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_labels, dropout):
        super(Classifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels

        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Linear(self.embedding_dim, num_labels)

    def forward(self, inputs):
        tensor = self.dropout(inputs)
        tensor = self.ff(tensor)
        tensor = tensor.view(-1, self.num_labels)
        predict = softmax(tensor, dim=-1)
        return tensor, predict


data_df = pandas.read_csv('../data/ner/ner.csv')
sentences = data_df.text.tolist()
labels = data_df.labels.tolist()
sentences_filtered = []
labels_filtered = []
for text, label in zip(sentences, labels):
    if len(text.split()) == len(label.split()):
        sentences_filtered.append(text)
        labels_filtered.append(label)
    else:
        print(f'ignored: {text} {label}')

train_sentences = sentences_filtered[0:1500]
train_labels = labels_filtered[0:1500]
train_labels = [l.split() for l in train_labels]

test_sentences = sentences_filtered[1500:2000]
test_labels = labels_filtered[1500:2000]
test_labels = [l.split() for l in test_labels]

labels2idx = get_label2id_vocab(train_labels)
idx2labels = {labels2idx[key]: key for key in labels2idx}

train_labels = get_label_ids(train_labels, labels2idx)
test_labels = get_label_ids(test_labels, labels2idx)

encoder = SentenceTransformer('quora-distilbert-multilingual')

train_embeddings = encoder.encode(train_sentences,
                                  output_value='token_embeddings',
                                  show_progress_bar=True,
                                  convert_to_tensor=False,
                                  device='cpu',
                                  )

test_embeddings = encoder.encode(test_sentences,
                                 output_value='token_embeddings',
                                 show_progress_bar=True,
                                 convert_to_tensor=False,
                                 device='cpu',
                                 )

tokenizer = encoder.tokenizer

# Most models have an initial BOS/CLS token, except for XLNet, T5 and GPT2
begin_offset = 1
if type(tokenizer) == XLNetTokenizer:
    begin_offset = 0
if type(tokenizer) == T5Tokenizer:
    begin_offset = 0
if type(tokenizer) == GPT2Tokenizer:
    begin_offset = 0


def subword2word_embeddings(subword_embeddings, text):
    sentence_embeddings = []
    pooling = 'mean'
    for embedding, sentence in zip(subword_embeddings, text):
        start_sub_token = begin_offset
        token_embeddings = []
        for word in sentence.split():
            sub_tokens = tokenizer.tokenize(word)
            num_sub_tokens = len(sub_tokens)
            all_embeddings = embedding[start_sub_token:start_sub_token + num_sub_tokens]

            # Convert numpy embeddings to tensor
            all_embeddings = from_numpy(all_embeddings)

            if pooling == 'mean':
                final_embeddings = torch.mean(all_embeddings, dim=0)
            elif pooling == 'last':
                final_embeddings = all_embeddings[-1]
            else:
                final_embeddings = all_embeddings[0]

            token_embeddings.append(final_embeddings)
            start_sub_token += num_sub_tokens

        sentence_embeddings.append(torch.stack(token_embeddings))

    return sentence_embeddings


train_embeddings = subword2word_embeddings(train_embeddings, sentences_filtered)
test_embeddings = subword2word_embeddings(test_embeddings, test_sentences)
# This will create a array of pointers to variable length tensors
# np_array = numpy.asarray(sentence_embeddings, dtype=object)

model = Classifier(embedding_dim=768, num_labels=len(labels2idx), dropout=0.001)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for e in range(50):
    total_loss = 0
    for emb, label in zip(train_embeddings, train_labels):
        label = torch.LongTensor(label)
        optimizer.zero_grad()
        model_output, _ = model(emb)
        loss = loss_fn(model_output, label.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)

final_predictions = []
final_labels = []
for emb, label in zip(test_embeddings, test_labels):
    _, predict_prob = model(emb)
    predict_labels = torch.argmax(predict_prob, dim=-1)
    final_predictions.extend(predict_labels.tolist())
    final_labels.extend(label)

print(classification_report(final_predictions, final_labels))
