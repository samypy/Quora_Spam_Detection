# -*- coding: utf-8 -*-
"""Quora_base_LSTM_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZXTvwYmKF9LyKuumXw0Dul5GLneS8JrD
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model

df = pd.read_csv(r'/content/drive/My Drive/Quara Spam filter/train.csv')

df['target'].value_counts()

df_1 = df[df['target']==1]
df_1.reset_index(inplace=True)
df_0 = df[df['target']==0]
df_00 = df_0[0:df_1.shape[0]]
df_00.reset_index(inplace=True)
df_10 = pd.concat([df_1,df_00],axis=0)

df_10['target'].value_counts()

del df

train,test=train_test_split(df_10,test_size=0.2,random_state=2)

df_train,df_val=train_test_split(train,test_size=0.2,random_state=2)

x_train=df_train['question_text']
y_train=df_train['target']
x_val=df_val['question_text']
y_val=df_val['target']

import nltk
nltk.download('punkt')

sent_lens=[]
for sent in df_train['question_text']:
    sent_lens.append(len(word_tokenize(sent)))

max(sent_lens)

np.quantile(sent_lens,0.95)

max_len = 40

tok = Tokenizer(char_level=False,split=' ')

tok.fit_on_texts(x_train)

sequences_train = tok.texts_to_sequences(x_train)

vocab_len=len(tok.index_word.keys())

vocab_len

sequences_matrix_train = sequence.pad_sequences(sequences_train,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    
    layer = Embedding(vocab_len+1,500,input_length=max_len,
                      mask_zero=True)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])

sequences_test = tok.texts_to_sequences(x_val)
sequences_matrix_test = sequence.pad_sequences(sequences_test,
                                               maxlen=max_len)

from tensorflow.keras.callbacks import ModelCheckpoint
outputFolder = '/content/drive/My Drive/Quora_Base_Model/'
filepath=outputFolder+"/weights-{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True,
                             save_weights_only=False, 
                             mode='auto', save_freq='epoch')

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_loss', 
                          min_delta=0.01, patience=5,
                          verbose=1, mode='auto')

model.fit(sequences_matrix_train,y_train.values,batch_size=256,
          epochs=50,validation_data=(sequences_matrix_test,y_val.values),callbacks=[earlystop,checkpoint])

from keras.models import load_model
model = load_model('/content/drive/My Drive/Quora_Base_Model/weights-01-0.8945.h5')

sequences_test1 = tok.texts_to_sequences(test['question_text'])
sequences_matrix_test1 = sequence.pad_sequences(sequences_test1,
                                               maxlen=max_len)

predictions=model.predict(sequences_matrix_test1)

from sklearn.metrics import roc_auc_score

roc_auc_score(np.asarray(test['target']),predictions)

