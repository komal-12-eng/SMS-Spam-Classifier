import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Title of the Streamlit app
st.title('Email/SMS Spam Classifier')

# Description of the application
st.write("""
## Welcome to the Email/SMS Spam Classifier!
This app uses Natural Language Processing (NLP) techniques and a Machine Learning model to classify messages as spam or not spam.
Just enter your message below, and the classifier will predict whether it is spam.
""")

# Text area for user input
input_sms = st.text_area('Enter the message')

# Load the pre-trained TF-IDF vectorizer and the spam classification model
tfdif = pickle.load(open('vectorizer_spam.pkl', 'rb'))
model = pickle.load(open('model_spam.pkl', 'rb'))

def transform_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stem the tokens
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Process the input message
transform_sms = transform_text(input_sms)

# Vectorize the processed message
vector_input = tfdif.transform([transform_sms])

# Predict the class of the message
result = model.predict(vector_input)

# Display the prediction result when the button is clicked
if st.button('Predict'):
    if result == 1:
        st.write('### Result: Spam')
        # st.image('spam_image.png', width=200)  # Add an image for spam classification
    else:
        st.write('### Result: Not Spam')
        # st.image('not_spam_image.png', width=200)  # Add an image for non-spam classification
