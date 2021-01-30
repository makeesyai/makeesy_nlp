# model_name: 'distilbert-base-nli-mean-tokens'
# Sentences we want sentence embeddings for
sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
encodings = model.encode(sentences, convert_to_tensor=True)
print(type(encodings))
print(encodings.size())

scores = util.pytorch_cos_sim(encodings, encodings)
print(scores)
