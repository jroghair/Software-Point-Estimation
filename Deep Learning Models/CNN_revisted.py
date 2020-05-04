# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:22:33 2020

@author: jrogh
"""


import os
import pandas as pd
import numpy as np
import math

os.chdir('C:\\Users\\jrogh\\Documents\\SoftwareStoryPointsPrediction\\testVectorSystems_Glove')


df = pd.read_csv("mule_category.csv")
row, col = df.shape

y = df.iloc[:,1]
X = df.iloc[:,2:]

x_train, x_test = X.iloc[:math.floor(row*0.7),:].to_numpy(), X.iloc[math.floor(row*0.7):,:].to_numpy()
y_train, y_test = y.iloc[:math.floor(row*0.7)].to_numpy(), y.iloc[math.floor(row*0.7):,].to_numpy()

#Simple CNN Model 
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras import backend as K

batch_size = 32
embed = (col-2)#*batch_size
max_length = 50
vocab_size = X.to_numpy().max()+1

def model_cnn():
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(embed,))
    print(inp.shape)
    x = Embedding(vocab_size, 300, input_length=max_length)(inp)
    print(x.shape)
    #x = Embedding(50, 50, weights=[embedding_matrix])(inp)
    x = Reshape((embed, 300, 1))(x)
    print(x.shape)
    
    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed),
                                     kernel_initializer='he_normal', activation='relu')(x)
        print(conv.shape)
        maxpool_pool.append(MaxPool2D(pool_size=(max_length - filter_sizes[i] + 1, 1))(conv))
        
    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(4, activation="softmax")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

cnn_model = model_cnn()

#model_conv.add(Dense(16, activation='softmax'))
#model_conv.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn_model.fit(x_train, y_train, validation_split=0.3, batch_size=32, epochs = 20)

'''
#trouble shooting layer sizes
a = np.random.rand(32,50,251,36)
b = np.random.rand(32,49,251,36)
c = np.random.rand(32,48,251,36)
d = np.random.rand(32,46,251,36)

pool = [a,b,c,d]

z = Concatenate(axis=1)(pool)   
z.shape
z = Flatten()(z)
z = Dropout(0.5)(z)
z = Dense(3, activation="softmax")(z)
'''