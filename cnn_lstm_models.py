

import os
import nltk
import re
from nltk import sent_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.layers import Embedding
## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
import pandas as pd
print(tf.__version__)

# Import PyDrive and associated libraries.
# This only needs to be done once per notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz

downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('clean_moodle.csv')  
df = pd.read_csv('moodle.csv')
#print('Downloaded content "{}"'.format(downloaded.GetContentString()))

df.head()

nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
words = set(nltk.corpus.words.words())

#Unique story points
uq = df['storypoint'].unique()
print(sorted(uq))
print(len(uq))

#Most frequent words
#df=pd.DataFrame(r1,columns=['text'])
df_freq = df.description.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
df_freq = df_freq.sort_values(ascending=False)
df_freq.head(30).plot.bar()

#subset data for modeling
data = df.reset_index()[['description', 'storypoint']]

#one-hot encoding response variable
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
data['sp'] = lb.fit_transform(data['storypoint']).tolist()
data = data[['description', 'sp']]
data.head()

vocab_size = 2000
maxlen = 35
tokenizer = Tokenizer(num_words= vocab_size)
tokenizer.fit_on_texts(df['description']) #index of words with length vocab size, ordered in length of frequency
sequences = tokenizer.texts_to_sequences(df['description'])
data_sq = pad_sequences(sequences, maxlen=maxlen)



#change response to 3 categories
#1-4 = low
#5-13 = medium
#13+ = high
data = df.reset_index()[['description', 'storypoint']]
data.storypoint.value_counts()
#data.loc[data.storypoint <=4, 'low'] = 0

data.loc[(data.storypoint <= 4), 'storypoints_mod'] = 'low'
data.loc[(data.storypoint >13), 'storypoints_mod'] = 'high'
data.loc[((data.storypoint <=13) & (data.storypoint > 4)), 'storypoints_mod'] = 'medium'

#one hot encode again
data['sp'] = lb.fit_transform(data['storypoints_mod']).tolist()
data = data[['description', 'sp']]
data.head()

#model1
def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(vocab_size, 100, input_length=50))
    model_conv.add(Dropout(0.5))
    model_conv.add(Conv1D(128, 10, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(125))
    model_conv.add(Dense(16, activation='softmax'))
    model_conv.compile(loss='sparse_categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
    return model_conv

enc = [data['sp'][i].index(1) for i in range(1166)]
model_conv = create_conv_model()
model_conv.fit(data_sq, np.array(enc), validation_split=0.3, epochs = 15)

from keras.utils import to_categorical
enc = np.array(data['sp']).argmax(axis=0)

#model2
#vocab_size = 2000
def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(vocab_size, 75, input_length=50))
    model_conv.add(Dropout(0.4))
    model_conv.add(Conv1D(150, 10, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=8))
    model_conv.add(LSTM(50))
    model_conv.add(Dense(16, activation='softmax'))
    model_conv.compile(loss='sparse_categorical_crossentropy', optimizer='adam',    metrics=['accuracy'])
    return model_conv

enc = [data['sp'][i].index(1) for i in range(1166)]
model_conv = create_conv_model()
model_conv.fit(data_sq, np.array(enc), validation_split=0.3, epochs = 15)

#Model 3 - with 2D convolutions WIP
model_conv = Sequential()
model_conv.add(Embedding(vocab_size, 128, input_length=50)) #output dimensions: [batch_size, input_length, 64]
model_conv.add(Reshape((50, 128, 1))) #shape now: [none, 50, 128, 1] channels_last
model_conv.add(Conv2D(60, 3, 1, padding="valid", name="conv2D-first", activation="relu")) #output shape: [samples, 48, 126, 60]
model_conv.add(MaxPooling2D(pool_size=4))
model_conv.add(Conv2D(1, 1, 1, padding="valid", name="conv2D-second", activation="relu")) #output shape:
model_conv.add(MaxPooling2D(pool_size=2))
model_conv.add(Dropout(0.3))
model_conv.add(Reshape((6,15)))
model_conv.add(LSTM(128))
model_conv.add(Dense(16, activation='softmax'))
model_conv.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_conv.fit(data_sq, np.array(enc), validation_split=0.3, epochs = 20)

#Debugging - verify output shape of model for layers
input_array = np.random.randint(1000, size=(32, 50))
output_array = model_conv.predict(input_array)
print(output_array.shape)
#assert(output_array.shape == (32, 48, 126, 60))

#CNN Model 4
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPool2D, Concatenate
from tensorflow.keras.models import Model
embed_size = 300
def model_cnn(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(vocab_size, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(3, activation="softmax")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

cnn_model = model_cnn(emb_mtx)

#model_conv.add(Dense(16, activation='softmax'))
#model_conv.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
enc = [data['sp'][i].index(1) for i in range(1166)]
cnn_model.fit(data_sq, np.array(enc), validation_split=0.3, epochs = 20)

#Useful Links
#http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
#https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
#https://towardsdatascience.com/how-to-build-a-gated-convolutional-neural-network-gcnn-for-natural-language-processing-nlp-5ba3ee730bfb
#http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
#https://github.com/dennybritz/cnn-text-classification-tf
#https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
