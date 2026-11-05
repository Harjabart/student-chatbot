import os
import json
import re
import pickle
import numpy as np
import nltk

# Download NLTK resources FIRST before importing corpus modules
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stop_words
    ]
    return " ".join(cleaned_tokens)

def train_chatbot():
    with open('data/intents.json', 'r') as file:
        data = json.load(file)

    corpus = []
    labels = []

    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            cleaned_pattern = preprocess_text(pattern)
            if cleaned_pattern:
                corpus.append(cleaned_pattern)
                labels.append(tag)

    # Scikit-Learn TF-IDF setup (Max features 1500, unigrams and bigrams)
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    y = np.array(labels)

    # Train Logistic Regression Model
    clf = LogisticRegression(solver='lbfgs', max_iter=500, C=1.0)
    clf.fit(X, y)

    # Create models folder if missing
    os.makedirs('models', exist_ok=True)

    # Serialize vectorizer & model
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("Model and Vectorizer trained and saved successfully!")

if __name__ == '__main__':
    train_chatbot()