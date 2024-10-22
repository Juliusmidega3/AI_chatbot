import random
import json
import pickle # for serialization
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
intents = json.loads(open('intents.json').read())

# Initialize necessary lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each pattern in intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize each pattern into words
        words.extend(word_list)  # Add the tokenized and lemmatized words to the list
        documents.append((word_list, intent['tag']))  # Add the document with its tag

        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add the unique tags to the classes list

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))  # set eliminates duplicates, sorted turns it back into a list
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))

# MACHINE LEARNING
training = []
output_empty = [0] * len(classes)

# Create bag of words and output rows
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag of words array with the length equal to number of unique words
    bag = [1 if word in word_patterns else 0 for word in words]
    
    # Create output array (0 for each class except the current class)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert training data to numpy arrays
random.shuffle(training)

# Ensure all elements have consistent shapes
train_x = np.array([np.array(entry[0]) for entry in training])
train_y = np.array([np.array(entry[1]) for entry in training])

# Building the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0])))
model.add(Activation('softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5', hist)  # or use 'chatbot_model.h5' for HDF5 format

print('Model training completed and saved.')

