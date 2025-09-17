# CADL1: Preprocessing Steps
# Import libraries
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download("punkt_tab")  # punkt tables (new requirement in recent NLTK versions)
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text corpus
corpus = [
    "Major technology companies are investing heavily in artificial intelligence.",
    "People are posting about climate change and sustainable living on social media.",
    "Morning yoga and high-protein diets are trending among fitness enthusiasts."
]

# Tokenization (NLTK)
print("🔹 Tokenization")
for text in corpus:
    tokens = word_tokenize(text)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}\n")

# Stopword removal (NLTK)
stop_words = set(stopwords.words('english'))
print("🔹 Stopword Removal")
for text in corpus:
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.lower() not in stop_words]
    print(f"Filtered Tokens: {filtered}\n")

# Stemming (NLTK - PorterStemmer)
ps = PorterStemmer()
print("🔹 Stemming")
for text in corpus:
    tokens = word_tokenize(text)
    stemmed = [ps.stem(w) for w in tokens]
    print(f"Stemmed: {stemmed}\n")

# Lemmatization (NLTK - WordNetLemmatizer)
lemmatizer = WordNetLemmatizer()
print("🔹 Lemmatization")
for text in corpus:
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    print(f"Lemmatized: {lemmatized}\n")

# Lemmatization with spaCy
nlp = spacy.load("en_core_web_sm")
print("🔹 Lemmatization with spaCy")
for text in corpus:
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    print(f"spaCy Lemmas: {lemmas}\n")
