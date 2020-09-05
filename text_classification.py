import tensorflow
import keras
import numpy

# print("text classification is real")

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[20])

# dictioanry for mapping words to integer indices
word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# dictionary for mapping integers into words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(integer_text):
    return " ".join([reverse_word_index.get(i, "?") for i in integer_text])


print(decode_review(train_data[20]))
