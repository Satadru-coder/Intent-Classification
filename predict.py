import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

sentence = ["do i need a umbrella today?"]
sequences = tokenizer.texts_to_sequences(sentence)
#print(sequences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(model.predict(padded))
index= np.argmax(model.predict(padded), axis=1)
print(classes[index][0])