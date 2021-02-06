"""
Models:
1. 'distilbert-base-nli-mean-tokens'
2. 'quora-distilbert-multilingual' (multilingual)

Steps:
1. Create pytorch model for text classification (explain each step with running the intermediate code)
2. Add Testing with training data using random model

3. Add Optimizer (Adam) and Loss function (CrossEntropy)
4. Add training Loop with logging loss
5. Add final evaluation code
"""
from sentence_transformers import SentenceTransformer

# Sentences we want sentence embeddings for
sentences = [
    'This framework generates embeddings for each input sentence',  # 0
    'Sentences are passed as a list of string.',  # 0
    'The quick brown fox jumps over the lazy dog.',  # 1
    'The lazy dog is also jumping.',  # 1
    'The fox and the dog are playing.'  # 1
]

test_sentences = [
    'The sentence is used here has good embeddings.',  # 0
    'The boy is playing with the dog and jumping in joy.', # 1
    # 'यहाँ वाक्य का इस्तेमाल किया गया है जिसमें अच्छी एम्बेडिंग है।',  # 0 (MT Hindi Language)
    # 'डॉग्स के साथ खेलता लड़का और खुशी से उछलता।'  # 1 (MT Hindi Language)
]

encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
embedding = encoder.encode(sentences)
print(embedding.shape)

from sentence_transformers import util

scores = util.pytorch_cos_sim(embedding, embedding)
print(scores)
