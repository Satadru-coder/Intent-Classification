import numpy as np
import json
from sklearn.preprocessing import LabelBinarizer

vocab_size = 1000
embedding_dim = 100
max_length = 200
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 503


with open("dataset.json", 'r') as f:
    datastore = json.load(f)

#print(datastore)
sentences=[]
intent=[]
entity=[]

for item in datastore['common_examples']:
    sentences.append(item['text'])
    intent.append(item['intent'])
#print(text)
#print(intent)
encoder= LabelBinarizer()
encoder.fit_transform(intent)
intent = encoder.transform(intent)
#print(intent)
classes= encoder.classes_

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_intents = intent[0:training_size]
testing_intents = intent[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
#print(word_index)
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#print(training_padded[0])
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_intents = np.array(training_intents)
testing_padded = np.array(testing_padded)
testing_intents = np.array(testing_intents)
#print(testing_intents)
