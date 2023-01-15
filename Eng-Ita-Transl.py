import random
import tensorflow as tf
import string
import re
import string
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras

choice = input("\nIf you want to use the smaller dataset press 1(suggested), if you want to use the full dataset press 2:\n")
if choice == '2':
    file = open("itaFull/itaFull.txt", 'r', encoding = "utf8")
else:
    file = open("ita.txt", 'r', encoding = "utf8")
    
raw_data = []

for line in file:
    pos = line.find("CC-BY")
    line = line[:pos-1]

    # Split the data into english and Italian
    eng, ita = line.split('\t')
    ita = "[start] " + ita + " [end]"

    # form tuples of the data
    data = eng, ita
    raw_data.append(data)

file.close()

def convert(list):
    return tuple(list)

data = convert(raw_data)

#Visualization of three examples of translation
print("\n")
print("Visualization of three examples of translation\n")
for i in range(3):
    print(random.choice(data))
    print("\n")


random.shuffle(raw_data)
totalDataLen = len(raw_data)
num_val_samples = int(0.15 * len(raw_data))

num_train_samples = len(raw_data) -2 * num_val_samples
train_pairs = raw_data[:num_train_samples]
val_pairs = raw_data[num_train_samples:num_train_samples + num_val_samples]
test_pairs = raw_data[num_train_samples + num_val_samples:]

#pulizia caratteri speciali nel dataset per migliorare la compresione del modello 
strip_chars = " "
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

f"[{re.escape(strip_chars)}]"


#Tokenizzazione delle parole del dataset
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

vocab_size = 15000
sequence_length = 20

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization
)

#divisione in tuple delle due traduzioni
train_english_texts = [pair[0] for pair in train_pairs]
train_italian_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_italian_texts)

#creazione del dataset vero e proprio
batch_size = 64

def format_dataset(eng, ita):
    eng = source_vectorization(eng)
    ita = target_vectorization(ita)
    return ({
        "english": eng,
        "italian": ita[:, :-1],
    }, ita[:, 1:])

def make_dataset(pairs):
    eng_texts, ita_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    ita_texts = list(ita_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, ita_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

print("\n")
print("example Tokenization of some phrases")
print("\n")
print(list(train_ds.as_numpy_iterator())[50])

#Creazione Encoder del Transformer
class TransformerEncoder(layers.Layer):
    def  __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(). __init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask = mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input+ proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

#Crazione classe Decoder del Tranformer
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)

#Position Embedding
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

        #Parametri del tranformer

choice = input("\nIf you want to custom your parameters press 1, if you want to use the suggested ones press 2:\n")
if choice == '2':
    embed_dim = 256
    dense_dim = 2048
    num_heads = 8

    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="italian")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #stampa dei parametri
    print("\n")
    print("Structure of the model")
    print("\n")
    transformer.summary()

    #definizione delle epoche ed effettiva fit del tranformer
    opt = keras.optimizers.Adam(learning_rate=0.001)
    transformer.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    history = transformer.fit(train_ds, epochs=20, validation_data=val_ds)

else:
    
    embed_dim = 256
    dense_dim = 2048
    num_heads = 8

    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="italian")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    
    choice2 = "2.0"
    while float(choice2) < 0.09 or float(choice2) > 1.0:
        choice2 = input("\nChoose dropout between 0.1 and 0.99 (suggested 0.5):\n")
    x = layers.Dropout(float(choice2))(x)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #stampa dei parametri
    print("\n")
    print("Structure of the model")
    print("\n")
    transformer.summary()

    #definizione delle epoche ed effettiva fit del tranformer
    choice = input("\nIf you want to use Adam press 1, if you want to use RMSprop press 2:\n")
    if choice == '2':
        opt = keras.optimizers.RMSprop(learning_rate=0.001)
    else:
        opt = keras.optimizers.Adam(learning_rate=0.001)
                   
    choice = input("\nDecide the number of epoches you want to do (number suggested for good results 20):\n")
    transformer.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    history = transformer.fit(train_ds, epochs=int(choice), validation_data=val_ds)


#Stampa del plot accuracy and val_loss
print("\nPrint of the accuracy and loss's plot \n")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Stampa di 5 traduzione eseguite dal modello
print("\n")
print("Print of 5 translation made by the model")
print("\n")
ita_vocab = target_vectorization.get_vocabulary()
ita_index_lookup = dict(zip(range(len(ita_vocab)), ita_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = ita_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(5):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
