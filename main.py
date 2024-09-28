# Step 1: Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Step 2: Load the IMDB dataset word index and reverse mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 3: Load the pre-trained model (Ensure the model file is in the working directory)
# If you get an error loading the model, ensure the .h5 file is properly downloaded
model = load_model('simple_rnn_imdb.h5')

# Step 4: Helper Functions

# Function to decode reviews from their encoded format to human-readable text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input text (convert text into model-compatible format)
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 5: Streamlit app

# Setting up the Streamlit app title and input area
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input area for entering a movie review
user_input = st.text_area('Movie Review', '')

# When the user clicks the 'Classify' button, process the input and classify the sentiment
if st.button('Classify'):
    if user_input.strip():  # Check if the input is not empty
        preprocessed_input = preprocess_text(user_input)  # Preprocess the input text
        
        # Make prediction using the pre-trained RNN model
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the sentiment and the prediction score (probability)
        st.write(f'Sentiment: **{sentiment}**')
        st.write(f'Prediction Score: {prediction[0][0]:.2f}')
    else:
        st.write('Please enter a valid movie review.')
else:
    st.write('Awaiting your input.')

