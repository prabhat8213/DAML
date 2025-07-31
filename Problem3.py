import nltk
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.tree import Tree

# NLTK Part
input_text = "Barack Obama went as a prime minister of USA in the year of 2015. PM MODI is the prime minister of INDIA."
ner = ne_chunk(pos_tag(word_tokenize(input_text)))

nltk_named_entity = []
for subtree in ner:
    if isinstance(subtree, Tree):
        entity = " ".join([token for token, pos in subtree.leaves()])
        nltk_named_entity.append(entity)

print("NLTK Named Entities:", nltk_named_entity)

# spaCy Part
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(input_text)
spacy_named_entity = [ent.text for ent in doc.ents]

print("spaCy Named Entities:", spacy_named_entity)
