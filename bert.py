# import tensorflow as tf
# from transformers import BertTokenizer, TFBertForTokenClassification

# model_name = 'bert-base-multilingual-cased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = TFBertForTokenClassification.from_pretrained(model_name)

# text = "Ma armastan Pariisi"
# inputs = tokenizer(text, return_tensors="tf")

# outputs = model(inputs)


# predicted_labels = tf.argmax(outputs.logits, axis=-1).numpy()[0]

# # Map predicted labels back to original words
# word_labels = []
# for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])):
#     if token.startswith("##"):
#         word_labels[-1][0] += token[2:]
#     else:
#         word_labels.append([token, predicted_labels[i]])

# # Print original text with predicted labels
# for word, label_id in word_labels:
#     print(f"Word: {word}, Predicted Label: {label_id}")


from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

# Define a random sentence
random_sentence = "Replace me by any text you'd like."

# Tokenize and encode the random sentence
inputs = tokenizer(random_sentence, return_tensors='tf')

# Perform inference to get predictions
outputs = model(inputs)

# Extract predicted probabilities for each class (assuming sequence classification task)
predicted_probabilities = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]

# Print predicted probabilities for each class
print("Predicted probabilities:")
for i, prob in enumerate(predicted_probabilities):
    print(f"Class {i}: {prob:.4f}")
