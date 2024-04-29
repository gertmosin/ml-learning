import tensorflow as tf
import pandas as pd
# import tensorflow_datasets as datasets
import datasets
from transformers import BertTokenizer, TFBertForTokenClassification, BertTokenizerFast
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences


conll2003 = datasets.load_dataset("conll2003") 
# def read_conll(filename):
#     df = pd.read_csv(filename, sep=' ', header = None, keep_default_na= False, names = ['TOKEN', 'POS', 'CHUNK', 'NE'], quoting=3, skip_blank_lines= False)
#     df['SENTENCE'] = (df.TOKEN == '').cumsum()
#     return df[df.TOKEN != '']

# sentences = read_conll('/Users/gertmosin/Downloads/project-1-at-2024-04-18-16-07-29b51187.conll')


example_text = conll2003['train'][0]


tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

tokenized_input = tokenizer(example_text['tokens'], is_split_into_words=True)

print(tokenized_input)

tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])

print(tokens)

word_ids = tokenized_input.word_ids()

print(word_ids)

print(len(example_text['ner_tags']), len(tokenized_input['input_ids']))
