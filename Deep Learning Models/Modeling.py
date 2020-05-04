# -*- coding: utf-8 -*-
"""
Attention model code is based from https://www.kaggle.com/sermakarevich/hierarchical-attention-network
and is based on Yang et al. 2016 paper titled "Hierarchical Attention Networks for Document Classification"
@author: jrogh
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import initializers, regularizers,constraints
from tqdm import tqdm_notebook as tqdm
import tensorflow.keras.backend as K
#from textblob import TextBlob
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
import math

os.chdir('C:\\Users\\jrogh\\OneDrive\\Documents\\665A\\Project')
base_directory = 'C:\\Users\\jrogh\\OneDrive\\Documents\\665A\\Project\\'
directory = 'C:\\Users\\jrogh\\OneDrive\\Documents\\665A\\Project\\Data'
clean_directory = 'C:\\Users\\jrogh\\OneDrive\\Documents\\665A\\Project\\Clean_data'


#preprocessing
embed_size = 300
vocab_size = 2000
maxlen = 35
tokenizer = Tokenizer(num_words= vocab_size)
lb = LabelBinarizer() #for one-hot encoding of response

#Setup Embedding matrix
glove_embedding_index = load_glove_index()

#Iterate through datasets applying CNN and Attention models
for filename in os.listdir(clean_directory):
    print(filename)
    df = pd.read_csv(clean_directory + "\\" + filename)
    data = df.reset_index()[['description', 'storypoint']]
    
    data.loc[(data.storypoint <= 4), 'storypoints_mod'] = 'low'
    data.loc[(data.storypoint >13), 'storypoints_mod'] = 'high'
    data.loc[((data.storypoint <=13) & (data.storypoint > 4)), 'storypoints_mod'] = 'medium'
    data['sp'] = lb.fit_transform(data['storypoints_mod']).tolist()
    data = data[['description', 'sp']]
    enc = [data['sp'][i].index(1) for i in range(df.shape[0])]
    enc = np.asarray(enc)
    
    #data setup
    tokenizer.fit_on_texts(df['description'].astype(str)) #index of words with length vocab size, ordered in length of frequency
    sequences = tokenizer.texts_to_sequences(df['description'].astype(str))
    data_sq = pad_sequences(sequences, maxlen=maxlen)
    
    split = 0.7
    x_dim = data_sq.shape
    x_train, x_test = data_sq[:math.floor(x_dim[0]*0.7),:], data_sq[math.floor(x_dim[0]*0.7):,:]
    
    y_train, y_test = enc[:math.floor(x_dim[0]*0.7),], enc[math.floor(x_dim[0]*0.7):,]
    
    #create embedding glove matrix for words
    emb_mtx = create_glove(tokenizer.word_index, glove_embedding_index)
    cnn_model = basic_cnn(emb_mtx)
    
    #CNN Model training
    cnn_model.fit(x_train, y_train, validation_split=0.1, epochs = 15)
    loss, acc = cnn_model.evaluate(x_test, y_test)
    with open((base_directory + "DL-Model_Accuracy.txt"), 'a') as model:
             model.write(filename + " Basic-CNN-Model " + str(loss) + " " + str(acc) +"\n")
    
    #RNN Attention Model
    att_model = model_lstm_atten(emb_mtx)
    att_model.fit(x_train, y_train, validation_split=0.1, epochs = 15)
    loss, acc = att_model.evaluate(x_test, y_test)
    with open((base_directory + "DL-Model_Accuracy.txt"), 'a') as model:
             model.write(filename + " LSTM-Attention " + str(loss) + " " + str(acc) +"\n")
    
    

#create glove embeddings    
def load_glove_index():
    EMBEDDING_FILE = base_directory + 'glove.6B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8"))
    return embeddings_index
   
#Create glove embedding matrix for convolutional operations    
def create_glove(word_index,embeddings_index):
    emb_mean,emb_std = -0.005838499,0.48782197
    all_embs = np.stack(embeddings_index.values())
    embed_size = all_embs.shape[1]
    nb_words = min(vocab_size, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    count_found = nb_words
    for word, i in tqdm(word_index.items()):
        if i >= vocab_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] =  embedding_vector
        else:
            if word.islower():
                # try to get the embedding of word in titlecase if lowercase is not present
                embedding_vector = embeddings_index.get(word.capitalize())
                if embedding_vector is not None: 
                    embedding_matrix[i] = embedding_vector
                else:
                    count_found-=1
            else:
                count_found-=1
    print("Got embedding for ",count_found," words.")
    return embedding_matrix    
        
#Vanilla Convolutional Neural Network Model    
def basic_cnn(embedding_matrix):
    '''
    idea of how to implement a basic CNN Model to NLP task is from the blog:
    https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
    '''
    
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
        
#Dot product helper function for attention model
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
#Attention Model 
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(3, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    