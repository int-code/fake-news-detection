#
# Importing required modules
#
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


port_stem = PorterStemmer()

#
# Stemming and reducing texts to 50 words at most
#
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

    # Getting stopwords
    print("Downloading stopwords...")
    nltk.download('stopwords')
    print("Stopwords downloaded")

    print("Starting pre-processing...")

    # Reading data and pre-processing
    train_data = pd.read_csv('database/train.csv')
    train_data.fillna('', inplace=True)

    train_data['title'] = train_data['title'].apply(stem)
    print("Title stemmed")
    train_data['text'] = train_data['text'].apply(stem)
    print("Text stemmed")

    # Vectorizing title and text
    titlevectorizer = TfidfVectorizer(max_features=50)
    vectorizer = TfidfVectorizer(max_features=50)
    titlevectorizer.fit(train_data['title'].values)
    title = titlevectorizer.transform(train_data['title'].values)
    title = pd.DataFrame.sparse.from_spmatrix(title)
    print("Title vectorized")

    vectorizer.fit(train_data['text'].values)
    text = vectorizer.transform(train_data['text'].values)
    text = pd.DataFrame.sparse.from_spmatrix(text)
    print("Text vectorized")

    # merging both the dataframes into one
    text.columns = [x for x in range(50,100)]
    title = pd.merge(title, text, left_index=True, right_index=True)

    # splitting data into test and train
    X_train, X_test, Y_train, Y_test = train_test_split(title, train_data['label'], test_size = 0.2, stratify=train_data['label'], random_state=2)
    
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    print("Model finished training")
    
    # Storing for later use
    pickle.dump(titlevectorizer, open("pickles/titlevectorizer.pickle", "wb"))
    pickle.dump(vectorizer, open("pickles/vectorizer.pickle", "wb"))
    pickle.dump(model, open("pickles/model.pickle", "wb"))

    print("Model built sucessfully")
