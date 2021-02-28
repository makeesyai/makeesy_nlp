import numpy
import torch
from sentence_transformers import SentenceTransformer
from transformers import XLNetTokenizer, T5Tokenizer, GPT2Tokenizer

sentences = [
    'The quick brown fox jumps over the lazy dog .'
]

labels = ['O', 'O', 'O', 'Animal', 'O', 'O', 'O', 'O', 'Animal', 'O']
tokenized = ['The', 'quick', 'brown', 'f', '##ox', 'jump', '##s', 'over', 'the', 'la', '##zy', 'dog', '.']


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
for arr in np_array:
    print(arr.size())
