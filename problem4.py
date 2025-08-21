# PRACTICAL 4 : TF-IDF

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Example documents
documents = [ "apple boy cat", "apple cat dog", "dog egg fan" ]

# Preprocessing
preprocessed = []
for doc in documents:
    tokenization = word_tokenize(doc.lower())   # convert to lowercase + tokenize
    stop = [ps.stem(word) for word in tokenization if word not in stop_words]
    preprocessed.append(" ".join(stop))

print("Preprocessed Documents:")
print(preprocessed)

# TF-IDF
vector = TfidfVectorizer()
word_score = vector.fit_transform(preprocessed)

# Convert to DataFrame for readability
df = pd.DataFrame(word_score.toarray(), columns=vector.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(df)
