import spacy
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Initialize NLTK Stemmer
stemmer = PorterStemmer()

# Sample text
text = """Over 300 illegal Bangladeshi immigrants detained in Delhi in 3 days, deported: Police.
On Wednesday morning, 92 persons – 39 children and 22 women from the Outer district, 41 persons from 
the South district, and 28 from other districts — were picked up and handed over to the FRRO, who arranged 
a special flight for them from Hindon air base."""
# Process the text with spaCy
doc = nlp(text)

# Sentence Tokenization
print("Sentence Tokenization:")
for sent in doc.sents:
    print(sent)

# Word Tokenization
print("\nWord Tokenization:")
for token in doc:
    print(token.text)

# Stop Word Removal
print("\nAfter Stop Word Removal:")
filtered_tokens = [token for token in doc if not token.is_stop and token.is_alpha]
print([token.text for token in filtered_tokens])

# Lemmatization
print("\nLemmatization:")
for token in filtered_tokens:
    print(f"{token.text} --> {token.lemma_}")

# Stemming (using NLTK)
print("\nStemming:")
for token in filtered_tokens:
    print(f"{token.text} --> {stemmer.stem(token.text)}")
