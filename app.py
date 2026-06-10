import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="FAQ Chatbot", layout="wide")

faq = {
    "what is ai": "AI means Artificial Intelligence",
    "what is python": "Python is a programming language",
    "what is machine learning": "ML is a subset of AI",
    "what is streamlit": "Streamlit is a Python framework"
}

questions = list(faq.keys())
answers = list(faq.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

st.title("🤖 FAQ Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    if role == "user":
        st.write("🧑", msg)
    else:
        st.write("🤖", msg)

user_input = st.text_input("Ask your question")

if st.button("Send"):
    if user_input:

        st.session_state.chat.append(("user", user_input))

        with st.spinner("Thinking..."):
            time.sleep(1)

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)

        index = similarity.argmax()
        score = similarity[0][index]

        if score > 0.3:
            response = answers[index]
        else:
            response = "Sorry, I don't know the answer."

        st.session_state.chat.append(("bot", response))
