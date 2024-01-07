import time
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import tensorflow as tf

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import psutil



start_time = time.time()

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training , dtype=object)

# Split the data into training and validation sets
split_ratio = 0.8  # Adjust the split ratio as needed
split_index = int(len(training) * split_ratio)

train_data = training[:split_index]
validation_data = training[split_index:]

# Extract features (X) and labels (Y) for training and validation sets
train_x = list(train_data[:, 0])
train_y = list(train_data[:, 1])

validation_x = list(validation_data[:, 0])
validation_y = list(validation_data[:, 1])    
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. SGD with Nesterov accelerated gradient gives good results for this model
sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#  Experiment with data shuffling
# random.seed(42)  # Set a seed for reproducibility
# random.shuffle(training)
# model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")


# Experiment with different batch sizes and epochs
# batch_sizes = [8, 16, 32]
# epochs_values = [50, 100, 200]

# for batch_size in batch_sizes:
#     for epochs_value in epochs_values:
#         model.fit(np.array(train_x), np.array(train_y), epochs=epochs_value, batch_size=batch_size, verbose=0)
#         evaluation_metrics = model.evaluate(np.array(validation_x), np.array(validation_y), verbose=0)
#         print(f"Batch Size: {batch_size}, Epochs: {epochs_value}, Metrics: {evaluation_metrics}")


end_time = time.time()
# Calculate training duration
training_duration = end_time - start_time
print(f"Training Duration: {training_duration} seconds")


# Monitor CPU and memory usage
cpu_usage_percent = psutil.cpu_percent()
memory_usage_percent = psutil.virtual_memory().percent
print(f"CPU Usage: {cpu_usage_percent}% | Memory Usage: {memory_usage_percent}%")


