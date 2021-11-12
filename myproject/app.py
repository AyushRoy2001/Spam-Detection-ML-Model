import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10)

model = pickle.load(open('spam.pkl','rb'))



st.title('Spam Mail Predictor')

page_bg_img = '''
<style>
body {
background-image: url("https://images.app.goo.gl/iLCiyjKQ9DsmN4zCA");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
name = st.text_input('Full Name')
message = st.text_input('Message')

if st.button('Predict'):
    Message = [message]
    new_message = vectorizer.fit_transform(Message)
    predictions = model.predict(new_message)
    if predictions == 1:
       st.header("Spam Mail!!")
    else:
       st.header("Not a Spam.")   
