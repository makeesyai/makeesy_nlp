import sys
import warnings

import numpy
import pandas
import torch
from numpy import VisibleDeprecationWarning
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from torch import nn
from torch import from_numpy
from torch.nn.functional import softmax
from transformers import XLNetTokenizer, T5Tokenizer, GPT2Tokenizer

# Add these to silence the warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)


def get_label2id_vocab(data):
    label2idx = {'PAD': 0, 'UNK': 1}
    idx = len(label2idx)
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
            label_ids.append(labels2ids.get(l, labels2ids.get('UNK')))
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
        probs = softmax(tensor, dim=-1)
        return tensor, probs


class Batcher(object):
    def __init__(self, data_x, data_y, batch_size,
                 emb_dim, pad_id, pad_id_labels, seed=0):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.num_samples = data_x.shape[0]
        self.emb_dim = emb_dim
        self.indices = numpy.arange(self.num_samples)
        self.ptr = 0
        self.rnd = numpy.random.RandomState(seed)
        self.pad_id = pad_id
        self.pad_id_labels = pad_id_labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= self.num_samples:
            self.ptr = 0
            self.rnd.shuffle(self.indices)
            raise StopIteration
        else:
            batch_x = self.data_x[self.ptr: self.ptr + self.batch_size]
            batch_y = self.data_y[self.ptr: self.ptr + self.batch_size]
            self.ptr += self.batch_size

            lengths = [len(x) for x in batch_x]
            max_length_batch = max(lengths)

            paddings = torch.full((max_length_batch * self.emb_dim,),
                                  fill_value=self.pad_id,
                                  dtype=torch.float32,
                                  )
            padded_labels = self.pad_id_labels * numpy.ones((len(batch_y), max_length_batch))
            all_emb = list()
            for idx, embeddings in enumerate(batch_x):
                curr_len = embeddings.shape[0]
                padded_labels[idx][:curr_len] = batch_y[idx]

                all_emb += [emb for emb in embeddings]
                num_pads = max_length_batch - curr_len

                if num_pads > 0:
                    t = paddings[
                        :num_pads * self.emb_dim
                    ]
                    all_emb.append(t)
            batch_x_final = torch.cat(all_emb).view(
                [len(batch_x), max_length_batch, self.emb_dim]
            )

            batch_y_final = torch.LongTensor(padded_labels)

            return batch_x_final, batch_y_final


data_df = pandas.read_csv('../data/ner/ner.csv')
print(data_df.head())
sentences = data_df.text.tolist()
labels = data_df.labels.tolist()
sentences_filtered = []
labels_filtered = []
for text, label in zip(sentences, labels):
    if len(text.split()) == len(label.split()):
        sentences_filtered.append(text)
        labels_filtered.append(label)
    else:
        sys.stderr.write(f'Ignored example: {text} {label}\n')


labels_filtered = [l.split() for l in labels_filtered]
train_sentences = sentences_filtered[0:4500]
train_labels = labels_filtered[0:4500]

test_sentences = sentences_filtered[4500:5000]
test_labels = labels_filtered[4500:5000]

labels2idx = get_label2id_vocab(train_labels)
idx2labels = {labels2idx[key]: key for key in labels2idx}

train_labels = get_label_ids(train_labels, labels2idx)
test_labels = get_label_ids(test_labels, labels2idx)

# Models:
# SBERT: https://www.sbert.net/docs/pretrained_models.html
# Huggingface: https://huggingface.co/models
#
# DATA
# Download link: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus

# encoder = SentenceTransformer('quora-distilbert-multilingual')
encoder = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')

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

            if not isinstance(all_embeddings, torch.Tensor):
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

n_samples = len(train_embeddings)
_, emb_dim = train_embeddings[0].size()

pad_id = tokenizer.pad_token_id
pad_id_labels = labels2idx['PAD']

# dtype=object is important
batcher = Batcher(numpy.asarray(train_embeddings, dtype=object),
                  numpy.asarray(train_labels, dtype=object), 32, emb_dim,
                  pad_id=pad_id, pad_id_labels=pad_id_labels)

model = Classifier(embedding_dim=emb_dim, num_labels=len(labels2idx), dropout=0.01)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for e in range(30):
    total_loss = 0
    for batch in batcher:
        batch_x, batch_y = batch
        optimizer.zero_grad()
        model_output, _ = model(batch_x)
        loss = loss_fn(model_output, batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch:{e}, Loss: {total_loss/n_samples}')

final_predictions = []
final_labels = []
for emb, label in zip(test_embeddings, test_labels):
    _, probs = model(emb)
    predict_labels = torch.argmax(probs, dim=-1)
    final_predictions.extend(predict_labels.tolist())
    final_labels.extend(label)

# Recommend for reading
# https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
print(classification_report(final_predictions, final_labels))
