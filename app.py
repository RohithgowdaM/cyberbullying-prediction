from flask import Flask, request, jsonify, render_template
from joblib import load
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
nb_model = load('./models/nb.joblib')
lr_model = load('./models/lr.joblib')
dt_model = load('./models/dt.joblib')
rf_model = load('./models/rf.joblib')

# Load TF-IDF vectorizer
tfidf = load('tfidf_vectorizer.joblib')  # Ensure to save and load the same TF-IDF used during training

# Define preprocessing functions
leet_dict = {
    '@': 'a', '$': 's', '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '!': 'i'
}

def normalize_leetspeak(text):
    for symbol, replacement in leet_dict.items():
        text = text.replace(symbol, replacement)
    return text

def preprocess_text_advanced(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Normalize leetspeak
    text = normalize_leetspeak(text)

    # Remove URLs and mentions
    text = re.sub(r'http\\S+|@\\w+', '', text)

    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stop words
    tokens = [re.sub(r'[^a-zA-Z]', '', word) for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word]

    return " ".join(lemmatized_tokens)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input with a 'text' field
    input_text = data.get('text')
    if not input_text:
        return jsonify({"error": "No text provided"}), 400
    try:
        # Preprocess the input text
        processed_text = preprocess_text_advanced(input_text)

        # Transform the text using the loaded TF-IDF vectorizer
        text_vector = tfidf.transform([processed_text])
        print(text_vector)
        # Get predictions from all models
        predictions = {
            "NaiveBayes": nb_model.predict(text_vector)[0],
            "LogisticRegression": lr_model.predict(text_vector)[0],
            "DecisionTree": dt_model.predict(text_vector)[0],
            "RandomForest": rf_model.predict(text_vector)[0],
        }
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
