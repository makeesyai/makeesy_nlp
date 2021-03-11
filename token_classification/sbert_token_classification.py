import numpy
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import XLNetTokenizer, T5Tokenizer, GPT2Tokenizer


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


sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'The lazy dog is also jumping .',
    'The fox and the dog are playing .'
]

label2ids = {'O': 0, 'Animal': 1, 'Props': 2}
id2label = {0: 'O', 1: 'Animal', 2: 'Props'}

labels = [
    [0, 2, 2, 1, 0, 0, 0, 2, 1, 0],
    [0, 2, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
]

# encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
encoder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
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
            final_embeddings = torch.mean(all_embeddings, dim=0)
        elif pooling == 'last':
            final_embeddings = all_embeddings[-1]
        else:
            final_embeddings = all_embeddings[0]

        token_embeddings.append(final_embeddings)
        start_sub_token += num_sub_tokens

    sentence_embeddings.append(torch.stack(token_embeddings))

# This will create a array of pointers to variable length tensors
np_array = numpy.asarray(sentence_embeddings, dtype=object)

model = Classifier(embedding_dim=768, num_labels=4, dropout=0.01)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for e in range(20):
    total_loss = 0
    for emb, label in zip(np_array, labels):
        label = torch.LongTensor(label)
        optimizer.zero_grad()
        model_output = model(emb)
        loss = loss_fn(model_output, label.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)

for emb, label in zip(np_array, labels):
    model_output = model(emb)
    predict = torch.argmax(model_output, dim=-1)
    print(predict)
    print(label)
