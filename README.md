# Intent-Classification	
This is a intent classifier using tensorflow and keras. a project of Natural Language Processing	
I used a dataset of  intents and text. Model contains keras sequential and bidirectional LSTM layer. 	

Text used for the training falls under the six categories namely, Greet, Goodbye, GetWeather , Calculaotr , Calendar, Calendar_update each having nearly 200 sentences.

AT first gather all the dataset and then put them in a dict with tag "common examples". or if you want you can customize the dataset as you want.
I'm getting validation accuracy almost 95%-96%. hope this code will work for others. :)

# Requirements
```
  
import numpy
import json
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
```
