import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


print("Starting pre-processing...")
train_data = pd.read_csv('train.csv')
train_data.fillna('', inplace=True)

port_stem = PorterStemmer()

def stem(content):
    stemmed = re.sub('[^a-zA-Z ]', '', content)
    stemmed = stemmed.lower()
    stemmed = stemmed.split(" ")
    stemmed = [port_stem.stem(x) for x in stemmed if not x in stopwords.words('english')]
    stemmed = " ".join(stemmed)
    return stemmed

train_data['title'] = train_data['title'].apply(stem)
print("Title stemmed")
train_data['text'] = train_data['text'].apply(stem)
print("Text stemmed")

train_data.to_csv('modified_train.csv')