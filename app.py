import json
import re
import random
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load Serialized Model & Vectorizer
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/intents.json', 'r') as f:
    intents_data = json.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

CONFIDENCE_THRESHOLD = 0.05

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(w) for w in tokens if
         w not in stop_words
    ]
    return " ".join(cleaned_tokens)

def get_intent_response(user_query):
    cleaned_query = preprocess_text(user_query)
    
    if not cleaned_query.strip():
        return "I am not sure I understood that. Could you please rephrase your question?"

    features = vectorizer.transform([cleaned_query])
    probabilities = model.predict_proba(features)[0]
    max_idx = probabilities.argmax()
    confidence = probabilities[max_idx]
    predicted_tag = model.classes_[max_idx]

    # Confidence Threshold Check
    if confidence < CONFIDENCE_THRESHOLD:
        return "I'm sorry, I didn't quite understand that query. Could you please rephrase it or contact the administrative support office?"

    for intent in intents_data['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

    return "Thank you for reaching out. Please specify your query further."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'response': 'Please enter a valid message.'})

    bot_reply = get_intent_response(user_message)
    return jsonify({'response': bot_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)