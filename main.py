import tensorflow as tf
import pandas as pd
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.ops import array_ops

# Step 1: Download the data from Kaggle and put it in the data repository.
try:
    df = pd.read_json("./data/Sarcasm_Headlines_Dataset_v2.json", lines=True)
except Exception:
    print('Download the dataset from the kaggle https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm'
          '-detection '
          'and paste it in the data folder, download the version 2.')
    sys.exit()

print('maximum length of headline data is ', df.headline.str.split(' ').map(len).max())
print('minimum length of headline data is ', df.headline.str.split(' ').map(len).min())
print('mean of the lengths of the headline data is ', df.headline.str.split(' ').map(len).mean())

# Set the hyperparametes
vocab_size = 1000
embedding_dim = 16
oov_tok = "<OOV>"
batch_size = 64

# Creating an instance of tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, lower=True)
# Creates and updates the internal vocabulary based on the text.
tokenizer.fit_on_texts(df.headline)

# Add padding token.
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Transforms the sentences to integers
sentences_int = tokenizer.texts_to_sequences(df.headline)
labels = df.is_sarcastic.values


# Using generator for creating the dataset.
def generator():
    for i in range(0, len(sentences_int)):
        # creates x's and y's for the dataset.
        yield sentences_int[i], [labels[i]]


# Calling the from_generator to generate the dataset.
# Here output types and output shapes are very important to initialize.
# the output types are tf.int64 as our dataset consists of x's that are int as well as the labels that are int as well.
# The tensorshape for x is tf.TensorShape([None]) as the sentences can be of varied length.
# The tensorshape of y is tf.TensorShape([1]) as that consists of only the labels that can be either 0 or 1.
dataset = tf.data.Dataset.from_generator(generator, (tf.int64, tf.int64),
                                         (tf.TensorShape([None]), tf.TensorShape([1])))


# This function determines the length of the sentence.
# This will be used by bucket_by_sequence_length to batch them according to their length.
def _element_length_fn(x, y=None):
    return array_ops.shape(x)[0]


# These are the upper length boundaries for the buckets.
# Based on these boundaries, the sentences will be shifted to different buckets.
boundaries = [df.headline.map(len).max() - 850, df.headline.map(len).max() - 700, df.headline.map(len).max() - 500,
              df.headline.map(len).max() - 300, df.headline.map(len).max() - 100, df.headline.map(len).max() - 50,
              df.headline.map(len).max()]

# These defines the batch sizes for different buckets.
# I am keeping the batch_size for each bucket same, but this can be changed based on more analysis.
# As per the documentation - batch size per bucket. Length should be len(bucket_boundaries) + 1.
# https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length
batch_sizes = [batch_size] * (len(boundaries) + 1)

# Bucket_by_sequence_length returns a dataset transformation function that has to be applied using dataset.apply.
# Here the important parameter is pad_to_bucket_boundary. If this is set to true then, the sentences will be padded to
# the bucket boundaries provided. If set to False, it will pad the sentences to the maximum length found in the batch.
# Default value for padding is 0, so we do not need to supply anything extra here.
dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(_element_length_fn, boundaries,
                                                                       batch_sizes,
                                                                       drop_remainder=True,
                                                                       pad_to_bucket_boundary=True))


# Splitting the dataset for training and testing. - https://stackoverflow.com/a/58452268/7220545
def is_test(x, _):
    return x % 4 == 0


def is_train(x, y):
    return not is_test(x, y)


recover = lambda x, y: y

# Split the dataset for training.
test_dataset = dataset.enumerate() \
    .filter(is_test) \
    .map(recover)

# Split the dataset for testing/validation.
train_dataset = dataset.enumerate() \
    .filter(is_train) \
    .map(recover)

# Create keras squential model.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=[None]),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPool1D(),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Writing a early stopping callback. Monitoring the val_loss, if the val_loss stops to decrease
# the training will aborted.
callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

# Profile from batches 2 to 20
# Using the callback to profile the training.
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/',
                                             profile_batch='2, 20')

# Using the adam optimizer with default configuration.
# Using binary_crossentropy loss function as it is a binary classification task.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# printing the summary of the model.
model.summary()

# Train the model.
model.fit(train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE),
          validation_data=test_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE), epochs=35,
          callbacks=[callback_early_stopping, tb_callback])
