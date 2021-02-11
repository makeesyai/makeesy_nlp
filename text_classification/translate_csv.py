import pandas as pd

# data = pd.read_csv("../data/nlp-getting-started/train.csv", dtype=object)
data = pd.read_csv("../data/spam/SPAM text message 20170820 - Data.csv", dtype=object)

data.rename(columns={
    'Message': 'text',
    'Category': 'target',
    'message': 'text',
    'original': 'text_fr',
    'related': 'target'
},
    inplace=True)

data = data[['text', 'target']]
data.dropna(inplace=True)

# Uncomment the following code test the zero-shot Multilingual transfer
sentences = data.text.tolist()
num_samples = len(sentences)
start = 0
batch_size = 8
from easynmt import EasyNMT
model = EasyNMT('opus-mt')
# translator = GoogleTranslator(source='en', target='hi')
sentences_translated = list()
while True:
    batch = sentences[start:min(start + batch_size, num_samples)]
    batch_translated = model.translate_sentences(batch, source_lang='en', target_lang='hi')
    sentences_translated.extend(batch_translated)
    print(batch_translated)
    if start > num_samples:
        break
    start += batch_size

data['text_hi'] = sentences_translated
data.to_csv('../data/spam/data-en-hi.csv')
