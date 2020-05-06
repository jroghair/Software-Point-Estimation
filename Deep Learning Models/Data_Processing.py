# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:03:31 2020

@author: jrogh
"""

import pandas as pd
import os
import nltk
import re
from nltk import sent_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from tensorflow.keras.layers import Embedding
## Plotly
#import plotly.offline as py
#import plotly.graph_objs as go
#py.init_notebook_mode(connected=True)
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.manifold import TSNE

cwd = os.getcwd()
os.chdir(cwd)
directory = cwd + '\\Data'
save_directory = cwd + '\\Clean_data'

nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
words = set(nltk.corpus.words.words())

#df = pd.read_csv("moodle.csv")

for filename in os.listdir(directory):
    print(filename)
    df = pd.read_csv(directory + "\\" + filename)
    df["description"] = df["title"] + ' ' +  df["description"]
    #edit descriptions
    for d in range(df.shape[0]):
        text = df['description'][d]
        text = str(text)
        text = text.lower() #change to lowercase
        text = " ".join(w for w in nltk.wordpunct_tokenize(text) 
                if w in words and w.isalpha()) #remove non-english words
        text = re.sub('<[^<]+?>', '', text) #remove html tags
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #remove links
        text = text.strip('=')
        text = text.split(" ") 
        text = [w for w in text if w not in nltk.corpus.stopwords.words('english')]
        lem = nltk.stem.WordNetLemmatizer()
        text = [lem.lemmatize(i) for i in text]
        text = ' '.join(str(e) for e in text)
        df.iloc[d,2] = text #replace description in place in the dataframe

    df.to_csv(save_directory + "\\clean_"+filename)
    
    




