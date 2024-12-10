import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Prepare training data
training_sentences = []
training_labels = []
labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize each word
        word_list = nltk.word_tokenize(pattern)
        training_sentences.append([lemmatizer.lemmatize(word.lower()) for word in word_list])
        training_labels.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(training_labels)

# Create a vocabulary
all_words = sorted(set([word for sentence in training_sentences for word in sentence]))

# Create training data
X_train = np.array([[1 if word in sentence else 0 for word in all_words] for sentence in training_sentences])
y_train = to_categorical(encoded_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create a neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(all_words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model.h5')

# Function to predict the response
def predict_response(user_input):
    user_input = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(user_input)]
    input_data = np.array([[1 if word in user_input else 0 for word in all_words]])
    prediction = model.predict(input_data)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])

    for intent in intents['intents']:
        if intent['tag'] == tag[0]:
            return random.choice(intent['responses'])

    return random.choice(intents['intents'][-1]['responses'])  # Default response

# Chatbot loop
def chatbot():
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response = predict_response(user_input)
        print(f"Chatbot: {response}")

# Run the chatbot
chatbot()
