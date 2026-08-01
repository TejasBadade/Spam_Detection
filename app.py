# Import required libraries
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (runs once)
nltk.download('stopwords', quiet=True)

# ============================================================
# Load saved model and vectorizer from pickle files
# ============================================================

# Load the trained Naive Bayes model
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# ============================================================
# Text cleaning function - same as in the notebook
# Must be identical otherwise predictions will be wrong
# ============================================================

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove everything except letters and numbers
    text = re.sub(r'[^a-z0-9]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Join back into string
    return " ".join(words)

# ============================================================
# Prediction function
# ============================================================

def predict_spam(text):
    # Clean the input text
    cleaned = clean_text(text)
    
    # Convert to TF-IDF vector using already fitted vectorizer
    vector = tfidf.transform([cleaned]).toarray()
    
    # Get prediction from model
    prediction = model.predict(vector)[0]
    
    # Get probability scores for confidence display
    probability = model.predict_proba(vector)[0]
    
    return prediction, probability

# ============================================================
# Streamlit App Interface
# ============================================================

# Page configuration - sets browser tab title and icon
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# App title and description
st.title("🛡️ SMS Spam Detector")
st.markdown("Enter any SMS message below to check if it is **spam or not spam.**")
st.markdown("---")

# Text input box for the user
user_input = st.text_area(
    label="Enter SMS message here",
    placeholder="Type or paste your SMS message...",
    height=150
)

# Predict button
if st.button("🔍 Check Message", use_container_width=True):
    
    # Make sure user actually typed something
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    
    else:
        # Get prediction and probability
        prediction, probability = predict_spam(user_input)
        
        # Display result based on prediction
        if prediction == 1:
            st.error("🚨 SPAM DETECTED")
            st.markdown(f"**Spam probability: {probability[1]*100:.1f}%**")
        else:
            st.success("✅ NOT SPAM")
            st.markdown(f"**Ham probability: {probability[0]*100:.1f}%**")
        
        # Show what the cleaned text looks like
        with st.expander("See how the message was processed"):
            st.write("**Original message:**")
            st.write(user_input)
            st.write("**After cleaning:**")
            st.write(clean_text(user_input))

# Divider
st.markdown("---")

# Model information section at the bottom
st.markdown("### 📊 Model Information")

# Display three metrics side by side using columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Algorithm", value="Naive Bayes")

with col2:
    st.metric(label="Accuracy", value="97.8%")

with col3:
    st.metric(label="Training Data", value="5,572 SMS")
