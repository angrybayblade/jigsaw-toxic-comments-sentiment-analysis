from warnings import filterwarnings
filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np

import tensorflow as tf
import spacy as sp
import regex as re


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score

from tensorflow.keras import layers,optimizers,activations,losses,Sequential,utils

from tqdm import tqdm
try:
    from notifyme import notify
except:
    pass

SYMBOL_FILTER = re.compile("[!@#%:;,\.?\'\"]")
CLEAN_UP = re.compile("[^a-z]")
WHITESPCAE_FILTER = re.compile("  *")

def extract_words(x):
    x = " ".join(SYMBOL_FILTER.sub("",x.lower()).split())
    x = " ".join(CLEAN_UP.sub(" ",x).split())
    x = " ".join(WHITESPCAE_FILTER.sub(" ",x).split())
    return x

print ("[+] Loading Data")

df = pd.read_csv("./train.csv")
df['comment_text_clean'] = df.comment_text.apply(extract_words)

print ("[+] Vectorizing")

cv = CountVectorizer()
features = cv.fit_transform(df.comment_text_clean.values.reshape(-1))
labels = df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].copy()

X,x,Y,y = train_test_split(features,labels.toxic.values)

print ("[+] Training")

model = Sequential()
model = Sequential([
    layers.Dense(32, input_shape=(features.shape[1],)),
    layers.Activation('relu'),
    layers.Dense(2),
    layers.Activation('softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 100
epochs = 5

for epoch in range(epochs):
    print (f"Epoch : {epoch}")
    for i in tqdm(range(batch_size,X.shape[0],batch_size)):
        x_batch = X[i-batch_size:i].toarray()
        y_batch = Y[i-batch_size:i]
        model.fit(x_batch,y_batch,batch_size=batch_size,verbose=False)
        
    x_batch = X[i:].toarray()
    y_batch = Y[i:]
    model.fit(x_batch,y_batch)

del x_batch,y_batch

y_pred = []

for i in range(5000,x.shape[0],5000):
    y_pred += model.predict_classes(x[i-5000:i].toarray()).tolist()
    
y_pred += model.predict_classes(x[i:].toarray()).tolist()

notify.success()

print (accuracy_score(y_pred,y),recall_score(y_pred,y),precision_score(y_pred,y))



