import pandas

data_df = pandas.read_csv("../data/spam/data-en-hi-de-fr.csv")
data_df.dropna(inplace=True)
data_df.drop_duplicates(inplace=True)
data_df.rename(columns={
    "target": "labels",
}, inplace=True)
# data_df = data_df[["labels", "text", "text_hi", "text_de", "text_fr"]]
data_df = data_df[["labels", "text", "text_hi", "text_de"]]
data_df.to_csv("../data/spam/data-en-hi-de.csv", index=False)
