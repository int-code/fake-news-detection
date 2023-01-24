import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


port_stem = PorterStemmer()
model = LogisticRegression()
titlevectorizer = TfidfVectorizer(max_features=50)
vectorizer = TfidfVectorizer(max_features=50)


def stem(content):
    stemmed = re.sub('[^a-zA-Z ]', '', content)
    stemmed = stemmed.lower()
    stemmed = stemmed.split(" ")
    if len(stemmed)>50:
        stemmed=stemmed[:50]
    stemmed = [port_stem.stem(x) for x in stemmed if not x in stopwords.words('english')]
    stemmed = " ".join(stemmed)
    return stemmed

if __name__=="__main__":
    print("Downloading stopwords...")
    nltk.download('stopwords')
    print("Stopwords downloaded")
    print("Starting pre-processing...")
    train_data = pd.read_csv('train.csv')
    train_data.fillna('', inplace=True)

    train_data['title'] = train_data['title'].apply(stem)
    print("Title stemmed")
    train_data['text'] = train_data['text'].apply(stem)
    print("Text stemmed")

    # train_data.to_csv('modified_train.csv')
    # train = pd.read_csv('modified_train.csv')
    # train.fillna('', inplace=True)

    titlevectorizer.fit(train_data['title'].values)
    title = titlevectorizer.transform(train_data['title'].values)
    title = pd.DataFrame.sparse.from_spmatrix(title)
    # print(title)

    # print("hsdh")
    vectorizer.fit(train_data['text'].values)
    text = vectorizer.transform(train_data['text'].values)
    text = pd.DataFrame.sparse.from_spmatrix(text)
    text.columns = [x for x in range(50,100)]
    title = pd.merge(title, text, left_index=True, right_index=True)
    # print(title)

    X_train, X_test, Y_train, Y_test = train_test_split(title, train_data['label'], test_size = 0.2, stratify=train_data['label'], random_state=2)

    model.fit(X_train, Y_train)
