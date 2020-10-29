import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score

df = pd.read_csv(r'/content/drive/My Drive/Quara Spam filter/train.csv')
df.head(2)
df['target'].value_counts()
df_1 = df[df['target']==1]
df_1.reset_index(inplace=True)
df_0 = df[df['target']==0]
df_00 = df_0[0:df_1.shape[0]]
df_00.reset_index(inplace=True)
df_10 = pd.concat([df_1,df_00],axis=0)
df_10['target'].value_counts()

MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.3

# Download glove embeddings
#!unzip '/content/drive/My Drive/Quara Spam filter/glove.6B.zip'

embeddings_index = {}
with open('/content/glove.6B.300d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_10['question_text'])
sequences = tokenizer.texts_to_sequences(df_10['question_text'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = np.asarray(df_10['target'])

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
layer = LSTM(64)(embedded_sequences)
layer = Dense(256,name='FC1')(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(1,name='out_layer')(layer)
layer = Activation('sigmoid')(layer)

model = Model(inputs=sequence_input,outputs=layer)
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=1024,
          epochs=15,
          validation_data=(x_val, y_val))

model.save('/content/drive/My Drive/Quara Spam filter/model.h5')


