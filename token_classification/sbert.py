import numpy
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import XLNetTokenizer, T5Tokenizer, GPT2Tokenizer


sentences = [
    'This framework generates embeddings for each input sentence .',  # 0
    'Sentences are passed as a list of string .',  # 0
    'The quick brown fox jumps over the lazy dog .',  # 1
    'The lazy dog is also jumping .',  # 1
    'The fox and the dog are playing .'  # 1
]

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
            final_embeddings = torch.mean(all_embeddings, dim=0)
        elif pooling == 'last':
            final_embeddings = all_embeddings[-1]
        else:
            final_embeddings = all_embeddings[0]

        token_embeddings.append(final_embeddings)
        start_sub_token += num_sub_tokens

    sentence_embeddings.append(torch.stack(token_embeddings))

for arr in sentence_embeddings:
    print(arr.size())
