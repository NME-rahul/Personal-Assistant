import pandas as pd
import tensorflow as tf
from tensorflo.keras.models import load_models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensoeflow.keras.preprocessing.sequences import pad_sequences

input_sequences = ''
one_hot_labels = ''
max_length = ''
vocab_size = ''

def load_data1():
    paths=['C:/Users/arvin/OneDrive/Desktop/data/target.txt', 'C:/Users/arvin/OneDrive/Desktop/data/input.txt']; data=[]
    for path in paths:
        with open(path) as fp:
            while True:
                line = fp.readline()
                if line == '':
                    break
                data.append(line)
    return data

def load_data2():
    paths='C:/Users/arvin/OneDrive/Desktop/data/lyrics_generator_82.h5'
    df = pd.read_csv(path)
    return list(df['poems'])

def create_sequence():
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(lyrics)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    print(word_index)
    print(vocab_size)
    sequences = []
    for line in lyrics:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		sequences.append(n_gram_sequence)

    max_length = max([len(seq) for seq in sequences])
    sequences = np.array(pad_sequences(sequences, maxlen=max_length, padding='pre'))

    input_sequences, labels = sequences[:,:-1], sequences[:,-1]
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes = vocab_size)
    input_sequences = tf.convert_to_tensor(input_sequences, dtype=tf.int64)
    one_hot_labels = tf.convert_to_tensor(one_hot_labels, dtype=tf.int64)
    return tokenizer

def create_model(vocab_size, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length - 1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    model.summary()
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model

def train(model, epochs, input_sequences, one_hot_labels):
    histrory = model.fit(
        input_sequences, one_hot_labels,
        epochs = epochs,
        verbose = 1
        )
    return history

def generate_text(model, tokenizer, seed_text, next_word):
    for _ in range(next_word):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
	#predicted = np.argmax(model.predict(token_list)[0], axis=0)
	predicted_probs = model.predict(token_list)[0]
	predicted = np.random.choice([x for x in range(len(predicted_probs))], p=predicted_probs)
	output_word = ""
	for word, index in tokenizer.word_index.items():
	    if index == predicted:
		output_word = word
		break
	seed_text += " " + output_word
    return seed_text
