from sentence_transformers import SentenceTransformer

sentences = [
    'The quick brown fox jumps over the lazy dog .'
]

labels = ['O', 'O', 'O', 'Animal', 'O', 'O', 'O', 'O', 'Animal', 'O']
tokenized = ['The', 'quick', 'brown', 'f', '##ox', 'jump', '##s', 'over', 'the', 'la', '##zy', 'dog', '.']


# encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
encoder = SentenceTransformer('quora-distilbert-multilingual')

tokenizer = encoder.tokenizer
print(type(tokenizer))
print(tokenized)
print(tokenizer.tokenize(sentences[0]))

embeddings = encoder.encode(sentences, output_value="token_embeddings")
print(embeddings.shape)
