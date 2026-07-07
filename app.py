#spam detection using simple RNN

#dataset spam.csv

#import libraries

import os
import re
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#configurations
Model = "spam_model.keras"
tokenizer_path = "tokenizer.pickle"

max_words = 5000
max_len = 150

#clean text function
def clean_text(text):
    text=str(text).lower()
    text=re.sub(r'\[.*?\]', '', text)
    text=re.sub(r'https?://\S+|www\.\S+', '', text)
    text=re.sub(r'<.*?>+', '', text)
    return text.strip()

#train model
def train_model():
    #load dataset
    df=pd.read_csv('spam.csv',encoding='latin-1')

    df = df[['v1','v2']]
    df.columns = ['label','message']
    print(df.head())
    print(df['label'].value_counts())
    # convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    # clean sms
    df['message'] = df['message'].apply(clean_text)
    
    # tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['message'])
    
    # save tokenizer
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['message'])
    X = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    # encode labels
    y = df['label']
    print("x shape:",X.shape)
    print("y shape:",y.shape)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # build model
    model=Sequential()
    model.add(Embedding(max_words, 32, input_length=max_len))
    
    #simple RNN layer
    model.add(SimpleRNN(128))
    #hidden layer
    model.add(Dense(64,activation='relu'))
    #output layer
    model.add(Dense(1,activation='sigmoid'))
     
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    #train model
    history=model.fit(X_train,y_train,epochs=10,batch_size=64,validation_data=(X_test,y_test))
    #save model
    model.save(Model)
    
    #evaluate model
    loss,accuracy=model.evaluate(X_test,y_test)
    print("\naccuracy: {:.2f}%".format(accuracy*100))
    print("\nloss: {:.2f}".format(loss))
    #predictions
    predictions=model.predict(X_test)


if not os.path.exists(Model) or not os.path.exists(tokenizer_path):
    train_model()

# predictions

def predict_sms(message):
    model = load_model(Model)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    message = clean_text(message)
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')

    probability = model.predict(padded)[0][0]
    prediction = "Spam" if probability > 0.5 else "Ham"
    return prediction, probability

#streamlit app
st.title("Spam Detection App")
st.write("many to one RNN Example")
st.write("This app uses a simple RNN model to detect spam messages.")   

message=st.text_area("Enter your message here:")
if st.button("Predict"):
    prediction,probability=predict_sms(message)
    st.success(prediction)
    st.write(f"Probability: {probability:.2f}")