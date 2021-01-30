# model_name: 'distilbert-base-nli-mean-tokens'
# INSTALL: sentence-transformers
# Sentences we want sentence embeddings for
sentences = [
    'This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.',
    'The fox was running towards the jungle.',
    'The dogs are following the fox.'
]

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
encodings = model.encode(sentences, convert_to_tensor=True)
print(type(encodings))
print(encodings.size())

# scores = util.pytorch_cos_sim(encodings, encodings)
# print(scores)
