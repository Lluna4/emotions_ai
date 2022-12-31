import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
em = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
model = tf.keras.models.load_model("modelo")

sequence = tokenizer.texts_to_sequences("You are the best")
padded = pad_sequences(sequence, padding="post")
a = model.predict(padded)
b = []
for i in a:
    b.append(sum(i))

print(em[np.argmax(b)])

