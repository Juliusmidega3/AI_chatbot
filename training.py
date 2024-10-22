import random
import json
import pickle #serialization
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer # reduce the word to its base form (lemma)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
intents = json.loads(open('intents.json').read())
nltk.download('punkt_tab')

# Initialize necessary lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each pattern in intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize each pattern into words
        word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in ignore_letters]  # Lemmatize and remove punctuation
        words.extend(word_list)  # Add the tokenized and lemmatized words to the list
        documents.append((word_list, intent['tag']))  # Add the document with its tag

        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add the unique tags to the classes list

# Print documents for verification
print(documents)
