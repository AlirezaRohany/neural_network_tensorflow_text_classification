import tensorflow
import keras
import numpy

# print("text classification is real")

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

print(train_data[20])

# dictionary for mapping words to integer indices
word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# dictionary for mapping integers into words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# showing a integer encode review to human readable review
def decode_review(integer_text):
    return " ".join([reverse_word_index.get(i, "?") for i in integer_text])


print(decode_review(train_data[20]))
print(len(test_data[0]), len(test_data[1]))

# preprocessing our data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",
                                                       maxlen=250)

print(len(test_data[0]), len(test_data[1]))

# # creating the model
# model = keras.Sequential()
# model.add(keras.layers.Embedding(88000, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation="relu"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))
#
# print("\n \n", model.summary())
#
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#
# x_validation = train_data[:10000]
# x_train = train_data[10000:]
#
# y_validation = train_labels[:10000]
# y_train = train_labels[10000:]
#
# fit_model = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_validation, y_validation),
#                       verbose=1)
#
# results = model.evaluate(test_data, test_labels)
# print("evaluate result: ", results)

# model.save("model.h5")

model = keras.models.load_model("model.h5")


# # checking a single review
# review = test_data[0]
# predict = model.predict([[review]])
# print("review: ", "\n", decode_review(review))
# print("prediction: " + str(predict[0]))
# print("actual: ", str(test_labels[0]))


