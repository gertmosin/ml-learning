import numpy as np
import tensorflow as tf
import estnltk
import pandas as pd

# Sample data (words and corresponding labels)
words = [
    ['Ma', 'armastan', 'Pariisi'],
    ['New', 'York'], 
    ['Tokyo', 'on', 'imeline'],
    ['Berliin', 'on', 'imeline'],
    ['Ma', 'armastan', 'reisida', 'Amsterdammi'],
    ['Kuhu', 'sina', 'tahaksid', 'reisida'],
    ['Parim', 'sihtkoht', 'maailmas', 'on', 'Prague'],
    ['Ou', 'Juss', 'anna', 'ampsu'],
    ['Mattias', 'lähme', 'teeme', 'Hollandis', 'peale'],
    ['Me', 'oleme', 'Berliinis'],
    ['Palun', 'tuvasta', 'vähemalt', 'Berliin', 'ära', "palun"],
    ['Tegelikult', 'vajab', 'see', 'päris', 'palju', 'andmeid'],
    ['Kõige', 'paremad', 'andmed', 'asuvad', 'USAs'],
    ['Ma', 'ausalt', 'ei', 'jaksa', 'enam'],
    ['Palun', 'päästa', 'mind', 'sellest', 'simulatsioonist'],
    ['Berliin', ',' ,'Tokyo', ',', 'Sydney', 'ja', 'Prague', 'on', 'minu', 'lemmikud', 'linnad'],
    ['London', 'on', 'kaunis', 'linn'],
    ['Ma', 'elasin', 'Pariisis', 'nüüd'],
    ['Eiffeli', 'torn', 'asub', 'Prantsusmaal'],
    ['Rooma', 'on', 'tuntud', 'oma', 'ajaloo', 'poolest'],
    ['Ma', 'tahan', 'kunagi', 'külastada', 'Veneetsiat'],
    ['Barcelona', 'on', 'kuulus', 'oma', 'arhitektuuri', 'poolest'],
    ['Istanbul', 'ühendab', 'Euroopat', 'ja', 'Aasiat'],
    ['Hiina', 'Müür', 'on', 'ajalooline', 'vaatamisväärsus'],
    ['Sydney', 'Ooperimaja', 'on', 'ikooniline', 'ehitis'],
    ['Ma', 'unistan', 'Põhjapooluse', 'valguse', 'nägemisest', 'Norras'],
    ['Rio', 'de', 'Janeiro', 'on', 'tuntud', 'oma', 'karnevali', 'poolest'],
    ['Giza', 'püramiidid', 'on', 'Egiptuses'],
    ['Mount', 'Everest', 'on', 'kõrgeim', 'mägi', 'maailmas'],
    ['Tallinn', 'on', 'Eesti', 'pealinn'],
    ['Tartu', 'on', 'Eesti', 'teine', 'suurem', 'linn'],
    ['Eesti', 'asub', 'Balti', 'mere', 'ääres'],
    ['Narva', 'asub', 'Eesti', 'idaosas'],
    ['Pärnu', 'on', 'tuntud', 'oma', 'ranniku', 'ja', 'kuurorti', 'poolest'],
    ['Eesti', 'elanike', 'arv', 'on', 'umbes', '1.3', 'miljonit'],
    ['Tallinna', 'lennujaam', 'on', 'suurim', 'lennujaam', 'Eestis'],
    ['Eesti', 'keel', 'kuulub', 'ugri-soome', 'keelte', 'hulka'],
    ['Tartu', 'ülikool', 'on', 'vanim', 'ülikool', 'Eestis'],
    ['Tallinna', 'vanalinn', 'on', 'UNESCO', 'maailmapärandi', 'nimekirjas'],

]
labels = [
    ['O', 'O', 'B-LOC'], 
    ['B-LOC', 'I-LOC'], 
    ['B-LOC', 'O', 'O'],
    ['B-LOC', 'O', 'O'],
    ['O', 'O', 'O', 'B-LOC'],
    ['O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'B-LOC'],
    ['O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'B-LOC', 'O'],
    ['O', 'O', 'B-LOC'],
    ['O', 'O', 'O', 'B-LOC', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'B-LOC'],
    ['O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'B-LOC', 'O', 'B-LOC', 'O', 'B-LOC', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'O', 'O'],
    ['O', 'O', 'B-LOC', 'O'],
    ['O', 'O', 'O', 'B-LOC'],
    ['B-LOC', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'B-LOC'],
    ['B-LOC', 'O', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'B-LOC'],
    ['B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'B-LOC'],
    ['O', 'O', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'B-LOC', 'O'],
    ['B-LOC', 'O', 'B-LOC', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'B-LOC', 'O'],
    ['B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-LOC'],
    ['B-LOC', 'O', 'O', 'O', 'O', 'O'],
    ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-LOC'],
    ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'O'],
]

# def read_conll(filename):
#     df = pd.read_csv(filename, sep=' ', header = None, keep_default_na= False, names = ['TOKEN', 'POS', 'CHUNK', 'NE'], quoting=3, skip_blank_lines= False)
#     df['SENTENCE'] = (df.TOKEN == '').cumsum()
#     return df[df.TOKEN != '']

# sentences = read_conll('/Users/gertmosin/Downloads/project-1-at-2024-04-18-16-07-29b51187.conll')
# words = [list(sentences['TOKEN'].values)]
# labels = [list(sentences['NE'].values)]


print(words)
print(labels)
# Create vocabulary and label dictionaries
word2idx = {'<pad>': 0, '<unk>': 1}
label2idx = {'<pad>': 0}

# Add all unique labels to the label dictionary
for sent_labels in labels:
    for label in sent_labels:
        if label not in label2idx:
            label2idx[label] = len(label2idx)

for sent_words in words:
    for word in sent_words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

# Find the maximum sequence length for words and labels
max_word_length = max(len(sent) for sent in words)
max_label_length = max(len(sent) for sent in labels)

# Convert words to indices and pad sequences
word_indices = []
for sent_words in words:
    word_indices.append([word2idx.get(word, 1) for word in sent_words] + [0] * (max_word_length - len(sent_words)))

# Convert labels to indices and pad sequences
label_indices = []
for sent_labels in labels:
    label_indices.append([label2idx[label] for label in sent_labels] + [0] * (max_label_length - len(sent_labels)))


# Define model parameters
vocab_size = len(word2idx)
embedding_dim = 100
hidden_dim = 100
num_labels = len(label2idx)
batch_size = 16
epochs = 15

# print('S6nad blj2d')
# print(word_indices)
# print('labels blj2d')
# print(label_indices)
# Define the LSTM model

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_word_length,)),
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_dim, return_sequences=True)),
    tf.keras.layers.Dense(num_labels, activation='softmax')
])
# model.add(tf.keras.layers.Dropout(0.2))
# inputs = tf.keras.Input(shape=(embedding_dim,))
# x = tf.keras.layers.Dense(embedding_dim, activation='relu')(inputs)
# x = tf.keras.layers.Dense(embedding_dim, activation='relu')(x)
# outputs = tf.keras.layers.Dense(num_labels, activation='softmax')(x)

# model = tf.keras.Model(inputs=inputs, outputs=outputs)


# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
# Prepare data for training
x_train = np.array(word_indices)
y_train = np.array(label_indices)

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Semi-supervised learning: Use the model to predict labels for unlabeled data

unlabeled_text = [
    ['Suvi', 'Eestis', 'on', 'ilus'],
    ['Kevad', 'toob', 'kaasa', 'roheluse'],
    ['Talv', 'on', 'külm', 'ja', 'lumerohke'],
    ['Sügis', 'toob', 'kaasa', 'värvikirevad', 'lehed'],
    ['Eestis', 'on', 'palju', 'ilusaid', 'looduskauneid', 'kohti'],
    ['Tallinn', 'on', 'populaarne', 'turismisihtkoht'],
    ['Nädalavahetusel', 'toimub', 'muusikafestival'],
    ['Suvehooaeg', 'toob', 'palju', 'turiste', 'Eestisse'],
    ['Tartu', 'ülikoolis', 'toimub', 'rahvusvaheline', 'konverents'],
    ['Eesti', 'köök', 'on', 'tuntud', 'oma', 'hõrkude', 'maitsete', 'poolest'],
    # estnltk.Text('Siin on minu lause, aga see lause on pikk').tag_layer(['words']),
    # estnltk.Text('Kas me Tallinnas klubisse kah lähme või?').tag_layer(['words']),
    # estnltk.Text("Kuule Mattias, kas sa oled käinud Amsterdamis või?").tag_layer(['words']),
    # estnltk.Text("Kui Praahasse lähete, teete seal peale ka või?").tag_layer(['words']),
    # estnltk.Text("Ma armastan Pariisi").tag_layer(['words']),
    # estnltk.Text("Mat OÜ  Registrikood: O3432432 Turu 34b  Bildly OÜ  Arve nr. 100045  Arve kuupäev: 18.11.2023 Maksetähtaeg: 02.12.2023 Registrikood: 16581720 KMKR: EE102539194 AS LHV PANK:  EE847700771008391286  BIC\/SWIFT: LHVBEE22  PROJEKT   PAKETT   KOGUS   HIND (NETO)  Kalender  18.11.2023 ... 18.12.2023   Basic   1 kuu   49.00 EUR  NETOSUMMA   KM (20%)   KOKKU  49.00 EUR   9.80 EUR   58.80 EUR  Bildly OÜ  Turu tn 2a 80010 Vändra alev, Põhja-Pärnumaa vald, Pärnu maakond Eesti  Aitäh! ").tag_layer(['words']),
]

unlabeled_data = []

# for text in unlabeled_text:
#     label=[]
#     for word in text.words:
#         label.append(word.text)
#     unlabeled_data.append(label)

for text in unlabeled_text:
    label=[]
    for word in text:
        label.append(word)
    unlabeled_data.append(label)


print(unlabeled_data)

# unlabeled_data = [['Berliin', 'on', 'tore'], ['Sydney', 'Opera', 'maja']]
unlabeled_word_indices = [[word2idx.get(word, 1) for word in sent] for sent in unlabeled_data]
# Find the maximum sequence length for the unlabeled data
max_unlabeled_seq_length = max(len(seq) for seq in unlabeled_word_indices)

# Pad sequences of unlabeled data to the maximum length
padded_unlabeled_word_indices = [seq + [0] * (max_unlabeled_seq_length - len(seq)) for seq in unlabeled_word_indices]

# Convert padded sequences to a NumPy array
x_unlabeled = np.array(padded_unlabeled_word_indices)


# Predict labels for unlabeled data
predicted_labels = model.predict(x_unlabeled)
predicted_label_indices = np.argmax(predicted_labels, axis=-1)


# Convert predicted label indices back to labels
idx2label = {idx: label for label, idx in label2idx.items()}
predicted_labels = [[idx2label[idx] for idx in sent] for sent in predicted_label_indices]

print(idx2label)

# Print predicted labels for unlabeled data
print("PREDICTID, PALUN:")
for sent, labels in zip(unlabeled_data, predicted_labels):
    print(sent)
    print(labels)
