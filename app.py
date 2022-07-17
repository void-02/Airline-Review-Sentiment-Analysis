import uvicorn
from fastapi import FastAPI
import numpy as np 
import pandas as pd 
import pickle
import pandas as pd
import numpy as np
import os
import nltk 
from nltk.corpus import stopwords
import requests
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from os.path import dirname, join, realpath
import joblib 
import regex as re


from data import txt
app = FastAPI(
    title="Airline review analysis API",
)


classifier = joblib.load("C:/Users/chira/MLAPI/classifier.pkl")

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    text = re.sub(r"\'s","", text)
    text = re.sub(' +',' ',text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)
    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)

    #text = "".join([c for c in text if c not in punctuation])

    if remove_stop_words:
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
        
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)


    return text 
    
    
    

    
@app.get('/')
def Introduction():
    return {"Detail":"This portal is designed to answer sentiments and emotions behind passengers sharing their flight experience. It also shows consumer satisfaction associated with different airlines."}


@app.post("/predict-review")
def predict_sentiment(review: str):

    cleaned_review = text_cleaning(review)

    prediction = classifier.predict([cleaned_review])
    output = int(prediction[0])
    
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    result = {"prediction": sentiments[output]}
    return result

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)




