import streamlit as st
import pickle
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Load the TED Talks data
df = pd.read_csv('JOINT_ted_video_transcripts_comments_stats.csv')

# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Combine title and transcript, then preprocess
df['details'] = df['title'] + ' ' + df['transcript']
df['details'] = df['details'].apply(preprocess_text)

# Load the pickled function
with open('Tedx_Talks_Recommendation_System\recommend_talks_with_sentiment.pkl', 'rb') as f:
    recommend_talks_with_sentiment = pickle.load(f)

# Streamlit app
st.title('TED Talks Recommender')

# Input fields
talk_content = st.text_input('Enter your talk content:')
comments = st.text_area('Enter comments (separate multiple comments with line breaks):')

# Button to generate recommendations
if st.button('Generate Recommendations'):
    # Preprocess input
    talk_content = preprocess_text(talk_content)
    comments = [preprocess_text(comment) for comment in comments.split('\n') if comment.strip()]
    
    # Get recommendations
    recommendations = recommend_talks_with_sentiment([talk_content], pd.Series(comments))
    
    # Display recommendations
    st.subheader('Recommended Talks:')
    for idx, talk in enumerate(recommendations):
        st.write(f'{idx+1}. {talk}')
