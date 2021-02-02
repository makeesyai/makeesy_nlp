# model_name: 'distilbert-base-nli-mean-tokens'
# INSTALL: sentence-transformers
# Sentences we want sentence embeddings for
sentences = [
    'This framework generates embeddings for each input sentence',  # 0
    'Sentences are passed as a list of string.',  # 0
    'The quick brown fox jumps over the lazy dog.',  # 1
    'The lazy dog is also jumping.',  # 1
    'The fox and the dog are playing.'  # 1
]

from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
embedding = encoder.encode(sentences)
print(embedding.shape)

from sentence_transformers import util

scores = util.pytorch_cos_sim(embedding, embedding)
print(scores)
