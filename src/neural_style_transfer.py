# -*- coding: utf-8 -*-

import pandas as pd
pd.set_option("display.max_colwidth", None)
import numpy as np

import tensorflow as tf
import transformers

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics.pairwise

import csv

# df_ = pd.read_csv("drive/MyDrive/Business/Künstliche Intelligenz Projekte/DeepMood/Twitter-US-Airline-Sentiment/Tweets.csv")
# df_ = df_[["text", "airline_sentiment", "airline_sentiment_confidence"]]
# df = df_[4000:5000]
# y = df.airline_sentiment
# labels = np.zeros(len(y))
# labels[y == "neutral"] = .5
# labels[y == "negative"] = 1.
# y = labels
# print("labels for sentiment:", y.shape)
# df

print("Loading dataset...")
df_ = pd.read_csv("data/training.1600000.processed.noemoticon.csv", header=None, names=["target", "ids", "date", "flag", "user", "text"], encoding="latin-1", quoting=csv.QUOTE_NONE, index_col=False)
df_ = df_[["text", "target"]]
df = df_[:100000]
df.dropna(inplace=True)
print(df)
# NOTE negative sentiment was decoded with 1 and positive with 0, so we want the sentiment to decrease

y = df.target
labels = np.zeros(len(y))
labels[y == 0] = 1.
y = labels
print("Labels for sentiment:", y.shape)

print("Loading model...")
# MODEL = "distilbert-base-uncased"
MODEL = "bert-base-uncased"
SEQ_LEN = 32
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
encoder = transformers.TFAutoModel.from_pretrained(MODEL, trainable=False)

text = "This is a Test Text!"
print("Text:", text)
input_ids = tokenizer.encode(text, return_tensors="tf", truncation=True, max_length=SEQ_LEN, padding="max_length")
print("input_ids:", input_ids)
decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("Decoded:", decoded)

vocab_size = tokenizer.vocab_size
print("Vocab_size:", vocab_size)

print("-" * 40)

input_decoder = tf.keras.layers.Input(shape=(SEQ_LEN, 768))
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, activation="relu", return_sequences=True, dropout=.25)) (input_decoder)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, activation="relu", return_sequences=True, dropout=.25)) (x)
x = tf.keras.layers.Dense(vocab_size, activation="sigmoid") (x)
decoder = tf.keras.Model(input_decoder, x, name="decoder")
print("Decoder:")
decoder.summary()

input_autoencoder = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
embedding = encoder(input_autoencoder).last_hidden_state[:]
x = embedding
x = decoder(x)
autoencoder = tf.keras.Model(input_autoencoder, x, name="autoencoder")
print("Autoencoder:")
autoencoder.summary()

texts = df.text.to_numpy()
input_ids = [tokenizer.encode(x, return_tensors="tf", truncation=True, max_length=SEQ_LEN, padding="max_length")[0] for x in texts]
input_ids = np.array(input_ids)
print("input_ids.shape =", input_ids.shape)

decoder_labelizer = sklearn.preprocessing.LabelBinarizer().fit(list(tokenizer.vocab.values()))

#######################################
def generator():
  for input_id in input_ids:
    X = input_id
    y = decoder_labelizer.transform(input_id).reshape((SEQ_LEN, vocab_size))
    yield X, y
  
dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.int64, tf.int64)).shuffle(100).batch(32)

#######################################
# target = decoder_labelizer.transform(input_ids.flatten()).reshape((input_ids.shape[0], SEQ_LEN, vocab_size))
# print("target.shape =", target.shape)
print("Fitting autoencoder...")
autoencoder.compile(optimizer="Adam", loss="categorical_crossentropy")
# autoencoder.fit(input_ids, target, epochs=10, batch_size=32)

autoencoder.fit(dataset, epochs=10)

# train_dataset_autoencoder = tf.data.Dataset.from_tensor_slices((input_ids, target)).shuffle(100).batch(32)
# autoencoder.fit(train_dataset_autoencoder, epochs=15)
#######################################

index = 1
print("Real:", tokenizer.decode(input_ids[index]))
pred = autoencoder.predict(input_ids[index : index + 1])
print("Model output:", pred.shape)
pred = decoder_labelizer.inverse_transform(pred.squeeze(0))
pred = tokenizer.decode(pred)
print("Pred:", pred)

print("-" * 40)

# # NOTE sentiment of text input
# inputs = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype=tf.int32)
# embedding = encoder(inputs).last_hidden_state
# x = embedding[:, 0]

# x = tf.keras.layers.Dense(100, activation="relu") (x)
# x = tf.keras.layers.Dense(1, activation="sigmoid") (x)

# sentiment_clf = tf.keras.Model(inputs, x)
# sentiment_clf.summary()

# NOTE sentiment of embedding input
embedding = tf.keras.layers.Input(shape=(768))

x = tf.keras.layers.Dense(10, activation="relu") (embedding)
x = tf.keras.layers.Dense(1, activation="sigmoid") (x)

sentiment_clf = tf.keras.Model(embedding, x, name="sentiment_classifier")
print("Sentiment model:")
sentiment_clf.summary()
sentiment_clf.compile(optimizer="Adam", loss="mse")

print("Creating embeddings as features for sentiment estimator...")
embeddings = encoder(input_ids).last_hidden_state[:,0]
print("Embeddings for sentiment:", embeddings.shape)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(embeddings.numpy(), y, test_size=.05)
print("Fitting sentiment model...")
sentiment_clf.fit(X_train, y_train, epochs=20, batch_size=32)

# TODO evaluate sentiment prediction
# y_pred = sentiment_clf.predict(X_test)

print("Changing the sentiment...")

TEXT = "The weather looks realy horrible"# NOTE expect output like: 'the weather could be better'
print("Original text:", TEXT)

# NOTE create embeddings
print("Creating feature embedding...")
input_ids_test = tokenizer.encode(TEXT, return_tensors="tf", truncation=True, max_length=SEQ_LEN, padding="max_length")
embedding_test = encoder(input_ids_test).last_hidden_state
print("embedding_test.shape:", embedding_test.shape)

# NOTE sentiment before
label_before = sentiment_clf.predict([embedding_test[:,0]])[0][0]
print(f"Sentiment: {label_before} [0=neg., 0,5=neut., 1=pos.]")

for layer in sentiment_clf.layers:
    layer.trainable = False
  
for layer in decoder.layers:
    layer.trainable = False

def loss_sentiment_content(embedding, content):

  def euclidean_distance(p, q):
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(q - p), axis=-1))

  def loss_content(embedding, content):
    # NOTE cosine similarity ranges from -1 (not similar) to 1 (similar) and we want the two embeddings to be similar, so we reverse its sign (1 - sim)
    sim = tf.keras.losses.cosine_similarity(embedding, content) + 1
    loss = 1 - (sim / 2)

    # loss = euclidean_distance(embedding, content)

    return loss

  def loss_sentiment(embedding):
    pred = sentiment_clf(embedding)[0]# TODO evaluate sentiment of decoded and again encoded embedding, so that changes, that have impact on sentiment, but not on decoders inperpretation, aren't a solution for optimization
    loss = 1 - pred
    return loss
  
  weight_sentiment = .1
  weight_content = .1
  
  loss = weight_content * loss_content(embedding, content) + weight_sentiment * loss_sentiment(embedding)
  return loss

opt = tf.keras.optimizers.Adam()
# opt = tf.keras.optimizers.Adam(learning_rate=.01, beta_1=.99, epsilon=1e-1)

embedding_ = embedding_test[0]

@tf.function()
def train_step(var):

  with tf.GradientTape() as tape:
    loss = loss_sentiment_content(var, embedding_)
  
  grad = tape.gradient(loss, var)
  opt.apply_gradients([(grad, var)])
  tf.clip_by_value(var, clip_value_min=0., clip_value_max=1.)

var = tf.Variable(embedding_)

for i in range(100):

  # if True:
  if i % 1 == 0:

    updated = var.numpy()

    sentiment = sentiment_clf.predict([updated])[0][0]
    print("Step {}:\tSentiment: {:.5f}".format(format(i, "03d"), sentiment))

    raw_text = decoder([np.expand_dims(updated, 0)]).numpy()
    raw_text = decoder_labelizer.inverse_transform(raw_text.squeeze(0))
    raw_text = tokenizer.decode(raw_text, skip_special_tokens=True)
    print("\t\tText:", raw_text)
    print()
  
  train_step(var)