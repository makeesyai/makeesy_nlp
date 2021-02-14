import pandas as pd

data = pd.read_csv('../data/ner/ner_dataset.csv')

data = data.fillna(method='ffill')
print(data.tail())
