import numpy
import torch

sentences = [
    'This framework generates embeddings for each input sentence',  # 0
    'Sentences are passed as a list of string.',  # 0
    'The quick brown fox jumps over the lazy dog.',  # 1
    'The lazy dog is also jumping.',  # 1
    'The fox and the dog are playing.'  # 1
]

from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
embedding = encoder.encode(sentences,
                           output_value='token_embeddings',
                           convert_to_tensor=True)
print(embedding.size())
tokenizer = encoder.tokenizer
offset = 1
sentence_embeddings = []
pooling = 'first'
for index, emb in enumerate(embedding):
    start_sub_token = offset
    token_embeddings = []
    for word in sentences[index].split():
        sub_tokens = tokenizer.tokenize(word)
        num_sub_tokens = len(sub_tokens)
        all_embeddings = embedding[index][start_sub_token:start_sub_token + num_sub_tokens]
        start_sub_token += num_sub_tokens
        if pooling == 'first':
            final_embeddings = torch.mean(all_embeddings, dim=0)
        elif pooling == 'last':
            final_embeddings = all_embeddings[0]
        else:
            final_embeddings = all_embeddings[0]
        token_embeddings.append(final_embeddings)
    sentence_embeddings.append(torch.stack(token_embeddings))

# This will create a array of pointers to variable length tensors
np_array = numpy.asarray(sentence_embeddings, dtype=object)
for arr in np_array:
    print(arr.size())
