# CADL4: Topic Modeling with LDA
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Sample text corpus
corpus = [
    "AI and machine learning are revolutionizing healthcare.",
    "Renewable energy sources are key to fighting climate change.",
    "The stock market fluctuates based on economic and political news.",
    "Yoga and meditation are becoming popular for mental health.",
    "Mars missions are advancing space exploration and technology."
]

# Preprocessing
stop_words = stopwords.words('english')
texts = [[word.lower() for word in doc.split() if word.lower() not in stop_words] for doc in corpus]

# Dictionary and Corpus
dictionary = corpora.Dictionary(texts)
doc_term_matrix = [dictionary.doc2bow(text) for text in texts]

# LDA Model
lda_model = gensim.models.LdaModel(doc_term_matrix, num_topics=3, id2word=dictionary, passes=15)

# Print topics
print("🔹 Topics Identified")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx}: {topic}")

# Visualization with pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Prepare visualization
vis = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)

# Save to HTML instead of trying to display in notebook
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("✅ Visualization saved as lda_visualization.html. Open this file in a browser.")
vis = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)
vis
