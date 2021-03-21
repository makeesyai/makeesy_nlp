import torch
from sentence_transformers import SentenceTransformer
from torch import nn
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


sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'The lazy dog is also jumping .',
    'The fox and the dog are playing .'
]

labels = [
    ['O', 'Props', 'Props', 'Animal', 'O', 'O', 'O', 'Props', 'Animal', 'O'],
    ['O', 'Props', 'Animal', 'O', 'O', 'O', 'O'],
    ['O', 'Animal', 'O', 'O', 'Animal', 'O', 'O', 'O']
]

labels2idx = get_label2id_vocab(labels)
idx2labels = {labels2idx[key]: key for key in labels2idx}

labels = get_label_ids(labels, labels2idx)
print(labels)

encoder = SentenceTransformer('quora-distilbert-multilingual')
embeddings = encoder.encode(sentences,
                            output_value='token_embeddings',
                            convert_to_tensor=True)

tokenizer = encoder.tokenizer

# Most models have an initial BOS/CLS token, except for XLNet, T5 and GPT2
begin_offset = 1
if type(tokenizer) == XLNetTokenizer:
    begin_offset = 0
if type(tokenizer) == T5Tokenizer:
    begin_offset = 0
if type(tokenizer) == GPT2Tokenizer:
    begin_offset = 0

sentence_embeddings = []
pooling = 'mean'
for embedding, sentence in zip(embeddings, sentences):
    start_sub_token = begin_offset
    token_embeddings = []
    for word in sentence.split():
        sub_tokens = tokenizer.tokenize(word)
        num_sub_tokens = len(sub_tokens)
        all_embeddings = embedding[start_sub_token:start_sub_token + num_sub_tokens]

        if pooling == 'first':
            final_embeddings = all_embeddings[0]
        elif pooling == 'last':
            final_embeddings = all_embeddings[-1]
        else:
            final_embeddings = torch.mean(all_embeddings, dim=0)

        token_embeddings.append(final_embeddings)
        start_sub_token += num_sub_tokens

    sentence_embeddings.append(torch.stack(token_embeddings))

for arr in sentence_embeddings:
    print(arr.size())


model = Classifier(embedding_dim=768, num_labels=len(labels2idx), dropout=0.01)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for e in range(50):
    total_loss = 0
    for emb, label in zip(sentence_embeddings, labels):
        label = torch.LongTensor(label)
        optimizer.zero_grad()
        model_output, _ = model(emb)
        loss = loss_fn(model_output, label.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)

for emb, label in zip(sentence_embeddings, labels):
    _, predict_prob = model(emb)
    predict_labels = torch.argmax(predict_prob, dim=-1)
    print(predict_labels.tolist())
    print(label)
