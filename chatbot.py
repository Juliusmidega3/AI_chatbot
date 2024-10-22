import random
import json
import pickle  # for serialization
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))

# Load the trained model
model = load_model('chatbot_model.h5')

# Clean up the sentence by tokenizing and lemmatizing it
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert a sentence into a bag-of-words representation
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent class based on the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

# Get a response based on the predicted intent
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "I'm sorry, I don't understand that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Main loop for interacting with the chatbot
print("Go! The bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
