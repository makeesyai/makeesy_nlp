from torch import nn, optim
from transformers import AutoTokenizer, AutoModel
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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
labels = torch.tensor([0, 0, 1])

# Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
num_sentence, embedding_dim = sentence_embeddings.size()

classifier = Classifier(embedding_dim, 2, dropout=0.01)
optimizer = optim.Adam(classifier.parameters())
loss_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    predict = classifier(sentence_embeddings)
    print(torch.argmax(predict, dim=-1))

for e in range(5):
    optimizer.zero_grad()
    logits = classifier(sentence_embeddings)
    loss = loss_fn(logits, labels)
    print(loss)
    loss.backward()
    optimizer.step()

test_sentences = ['The sentence is used here has good embeddings.',
                  'The boy playing with the Dogs and jumping with joy.']
test_labels = [0, 1]

# Tokenize sentences
encoded_test_input = tokenizer(test_sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_test_output = model(**encoded_test_input)
# Perform pooling. In this case, mean pooling
test_sentence_embeddings = mean_pooling(model_test_output, encoded_test_input['attention_mask'])

with torch.no_grad():
    predict = classifier(test_sentence_embeddings)
    print(torch.argmax(predict, dim=-1))
