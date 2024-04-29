# %matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus.reader import conll
from sklearn.model_selection import train_test_split

np.random.seed(0)
plt.style.use('ggplot')

from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow_datasets as tfds
print('Tensorflow version:', tf.__version__)
print('GPU detected:', tf.config.list_physical_devices('GPU'))

def read_conll(filename):
    df = pd.read_csv(filename, sep=' ', header = None, keep_default_na= False, names = ['TOKEN', 'POS', 'CHUNK', 'NE'], quoting=3, skip_blank_lines= False)
    df['SENTENCE'] = (df.TOKEN == '').cumsum()
    return df[df.TOKEN != '']

sentences = read_conll('/Users/gertmosin/Downloads/project-1-at-2024-04-18-16-07-29b51187.conll')

# print(sentences)

words = list(sentences['TOKEN'].values)
# words = list(set(sentences['TOKEN'].values))
# tags = list(set(sentences['NE'].values))
tags = list(sentences['NE'].values)

import tqdm
from tensorflow import keras

# print(words)
# print(tags)

def sentence_integrate(data):
    agg_func = lambda s: [(w, p, t) for w,p,t in zip(s['TOKEN'].values.tolist(), s['POS'].values.tolist(), s['NE'].values.tolist())]
    return data.groupby('SENTENCE').apply(agg_func).tolist()

data = sentence_integrate(sentences)

# print(data)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

# print(words)

# print(word2idx)
# print(tag2idx)
# reverse_tag_map={v: k for k, v in tag2idx.items()}
# encoded_tags=[[tag2idx[w] for w in tag] for tag in tags]

max_len = 64
embedding_dim=64
vocab_size=len(words)-1
lstm_units=64

X = [[word2idx[w[0]] for w in s] for s in data]
X = keras.preprocessing.sequence.pad_sequences(maxlen=max_len, sequences=X, padding='post', value=len(words)-1)

y = [[tag2idx[w[2]] for w in s] for s in data]
y = keras.preprocessing.sequence.pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag2idx['O'])


print(X)
# print(y)


# # split into test and train
# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# print('X TRAIN AND TRAINSASIDNASIDNAS')
# print(x_train)
# print(y_train)
# # x_val, x_val_test, y_val, y_val_test = train_test_split(x_test, y_test, test_size=0.2, random_state=1)

# # print(x_train[0].shape)
# # print(y_train[0].shape)

# print(X)

from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense, Reshape, Flatten
from keras.layers import InputLayer, TimeDistributed, SpatialDropout1D, Bidirectional


model = keras.models.Sequential()
model.add(InputLayer((max_len,)))
model.add(Embedding(input_dim=len(words), output_dim=max_len))
model.add(TimeDistributed(Dense(1,activation= 'sigmoid')))
# model.add(Reshape((-1, len(tags))))
# model.add(Flatten())

# print('input shape', model.input_shape)
# print('output shape', model.output_shape)

# # input_word = Input(shape= (max_len,))
# # model = Embedding(input_dim=vocab_size, output_dim= embedding_dim)(input_word)

# # model = LSTM(units=embedding_dim, return_sequences= True)(model)
# # out = TimeDistributed(Dense(len(tags), activation= 'softmax'))(model)
# # model = Model(input_word, out)

model.summary()

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

# # history = model.fit(x_train,np.array(y_train),validation_data=(x_test,np.array(y_test)),batch_size = 32,epochs = 3)


history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    epochs=3,
    batch_size=32,
)


# model.summary()
# model.save('/Users/gertmosin/Desktop/ML learning/model/my_model.keras')

# import tensorflowjs as tfjs

# # tfjs.converters.save_keras_model(model, '/Users/gertmosin/Desktop/ML learning/model/js')

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=32)
print("test loss: {} ".format(results[0]))
print("test accuracy: {} ".format(results[1]))

# i = np.random.randint(0, x_test.shape[0])
# print("This is sentence:",i)
# p = model.predict(np.array([x_test[i]]))
# p = np.argmax(p, axis=-1)

# print(p[0])

# print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
# print("-" *30)
# for w, true, pred in zip(x_test[i], y_test[i], p[0]):
#     print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

