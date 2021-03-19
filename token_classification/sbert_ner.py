import pandas
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch import from_numpy
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
            label_ids.append(labels2ids.get(l, 'O'))
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
        s, t = inputs.size()
        tensor = self.dropout(inputs)
        tensor = self.ff(tensor)
        return tensor.view(-1, self.num_labels)


data_df = pandas.read_csv('../data/ner/ner.csv')
train_sentences = data_df.text.tolist()[0:15]
labels = data_df.labels.tolist()[0:15]
labels = [l.split() for l in labels]

test_sentences = data_df.text.tolist()[15:20]
test_labels = data_df.labels.tolist()[15:20]
test_labels = [l.split() for l in test_labels]

labels2idx = get_label2id_vocab(labels)
idx2labels = {labels2idx[key]: key for key in labels2idx}

labels = get_label_ids(labels, labels2idx)
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


train_embeddings = subword2word_embeddings(train_embeddings, train_sentences)
test_embeddings = subword2word_embeddings(test_embeddings, test_sentences)
# This will create a array of pointers to variable length tensors
# np_array = numpy.asarray(sentence_embeddings, dtype=object)

model = Classifier(embedding_dim=768, num_labels=len(labels2idx), dropout=0.01)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for e in range(1):
    total_loss = 0
    for emb, label in zip(train_embeddings, labels):
        label = torch.LongTensor(label)
        optimizer.zero_grad()
        model_output = model(emb)
        loss = loss_fn(model_output, label.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)

for emb, label in zip(test_embeddings, test_labels):
    model_output = model(emb)
    predict = torch.argmax(model_output, dim=-1)
    print(predict.tolist())
    print(label)
