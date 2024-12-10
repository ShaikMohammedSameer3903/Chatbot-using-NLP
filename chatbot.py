import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

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

# Flatten the training sentences
flattened_sentences = [' '.join(sentence) for sentence in training_sentences]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(flattened_sentences, encoded_labels, test_size=0.2, random_state=42)

# Create a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to predict the response
def predict_response(user_input):
    user_input = lemmatizer.lemmatize(user_input.lower())
    prediction = model.predict([user_input])
    tag = label_encoder.inverse_transform(prediction)[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return random.choice(intents['intents'][-1]['responses'])  # Default response

# Chatbot loop
def chatbot():
    print("Chatbot is running! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = predict_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('wordnet')
    chatbot()
