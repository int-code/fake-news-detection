import pandas as pd
import pickle
from setup import stem

# Input the title and text of the news article
TITLE = ''
TEXT = ''


# Loading from storage
titlevectorizer = pickle.load(open("titlevectorizer.pickle", "rb"))
vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
model = pickle.load(open("model.pickle", "rb"))


# For passing input data through model
def check(news:dict):

    # Stemming text
    title = stem(news['title'])
    text = stem(news['text'])

    # Vectorizing
    title = titlevectorizer.transform([news['title']])
    title = pd.DataFrame.sparse.from_spmatrix(title)

    text = vectorizer.transform([news['text']])
    text = pd.DataFrame.sparse.from_spmatrix(text)

    # Merging
    text.columns = [x for x in range(50,100)]
    title = pd.merge(title, text, left_index=True, right_index=True)

    res = model.predict(title)
    return res


print(check({
    'title':TITLE,
    'text':TEXT
}))