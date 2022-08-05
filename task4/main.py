"""
Introduction to Machine Learning
Task 4: Image taste classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization, Dense, Lambda
from tensorflow.keras import optimizers, Sequential, Input, Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

#from google.colab import drive
#drive.mount('/content/gdrive')

# better save than sorry
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

USE_TRAINED = False

WIDTH = 224
HEIGHT = 224
CHANNELS = 3

EPOCHS = 8
LEARNING_RATE = 0.0002

NUM_EMBEDDINGS = 256
DROPOUT_RATE = 0.5
LOSS_MARGIN = 0.2

VALIDATION_SIZE = 0.2
VALIDATION_STEPS = 10

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
BUFFER_SIZE = 2056

#HANDOUT_PATH = "/content/gdrive/MyDrive/handout/"
HANDOUT_PATH = '/handout/'

# Seed for reproducability
seed = 42


def load_dataset(filename, augmenting):
    data = tf.data.TextLineDataset(filename)
    data = data.map(lambda triplet: load_triplets(triplet, augmenting),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return data


def load_triplets(triplet, augmenting):
    idx = tf.strings.split(triplet)
    anchor = process_image(HANDOUT_PATH + '/food/' + idx[0] + '.jpg', augmenting)
    positive = process_image(HANDOUT_PATH + '/food/' + idx[1] + '.jpg', augmenting)
    negative = process_image(HANDOUT_PATH + '/food/' + idx[2] + '.jpg', augmenting)
    return tf.stack([anchor, positive, negative], axis=0), 1


def process_image(filename, augmenting):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=CHANNELS)
    # Resize and scale images
    img = (tf.cast(img, tf.float32) - 127.5) / 127.5
    img = tf.image.resize(img, (HEIGHT, WIDTH))

    # Randomly flip images for augmentation
    if augmenting:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img


def create_prediction_model(model):
    dist_positive = tf.reduce_sum(tf.square(model.output[..., 0] - model.output[..., 1]), axis=1)
    dist_negative = tf.reduce_sum(tf.square(model.output[..., 0] - model.output[..., 2]), axis=1)
    predictions = tf.cast(tf.greater_equal(dist_negative, dist_positive), tf.int8)
    return Model(inputs=model.inputs, outputs=predictions)


class TripletLoss():
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=LOSS_MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, _, inputs):
        x0 = inputs[..., 0]
        x1 = inputs[..., 1]
        x2 = inputs[..., 2]

        # euclidian distance A-B
        diff1 = x0 - x1
        dist1 = tf.reduce_sum(tf.square(diff1), axis=1)

        # euclidian distance A-C
        diff2 = x0 - x2
        dist2 = tf.reduce_sum(tf.square(diff2), axis=1)

        # get the triplet loss
        mdist = self.margin + dist1 - dist2
        dist = tf.math.softplus(mdist)
        loss = tf.reduce_mean(dist)
        return loss


def accuracy(_, embeddings):
    dist_positive = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 1]), axis=1)
    dist_negative = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 2]), axis=1)
    return tf.reduce_mean(tf.cast(tf.greater_equal(dist_negative, dist_positive), tf.float32))


# =========
# Load data
# =========
with open(HANDOUT_PATH + 'train_triplets.txt', 'r') as file:
    triplets = [l for l in file.readlines()]

train_data, val_data = train_test_split(triplets, test_size=VALIDATION_SIZE, random_state=seed)
with open(HANDOUT_PATH + 'train_data.txt', 'w') as file:
    for e in train_data:
        file.write(e)
with open(HANDOUT_PATH + 'val_data.txt', 'w') as file:
    for e in val_data:
        file.write(e)

train_data = load_dataset(HANDOUT_PATH + 'train_data.txt', augmenting=True)
val_data = load_dataset(HANDOUT_PATH + 'val_data.txt', augmenting=False)

# Number of training and test data
with open(HANDOUT_PATH + 'train_data.txt', 'r') as file:
    num_train_data = sum(1 for line in file)
with open(HANDOUT_PATH + 'test_triplets.txt', 'r') as file:
    num_test_data = sum(1 for line in file)

# ==============
# Define a model
# ==============
basis_inputs = Input(shape=(3, HEIGHT, WIDTH, CHANNELS))
conv_net = MobileNetV2(include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))
# Do not train pretrained model
conv_net.trainable = False

# Additional layers for prediction
classifier = Sequential([
    GlobalAveragePooling2D(),
    Dropout(DROPOUT_RATE),
    #BatchNormalization(),
    Dense(NUM_EMBEDDINGS),
    Dropout(DROPOUT_RATE),
    #BatchNormalization(),
    Dense(NUM_EMBEDDINGS),
    Lambda(lambda x: tf.math.l2_normalize(x, axis=1))])

# Create feature vectors (embeddings)
anchor_feature = classifier(conv_net(basis_inputs[:, 0, ...]))
positive_feature = classifier(conv_net(basis_inputs[:, 1, ...]))
negative_feature = classifier(conv_net(basis_inputs[:, 2, ...]))
embeddings = tf.stack([anchor_feature, positive_feature, negative_feature], axis=-1)

# Siamese model
Loss = TripletLoss()
model = Model(inputs=basis_inputs, outputs=embeddings)
model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=[Loss.forward], metrics=[accuracy])

# ==================
# define dataloaders
# ==================
train_data = train_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).repeat().batch(TRAIN_BATCH_SIZE)
val_data = val_data.batch(TRAIN_BATCH_SIZE)

# ===========
# Train model
# ===========
model.fit(train_data, steps_per_epoch=int(np.ceil(num_train_data / TRAIN_BATCH_SIZE)),
          epochs=EPOCHS, validation_data=val_data, validation_steps=VALIDATION_STEPS)

# Predict labels
test_data = load_dataset(HANDOUT_PATH + 'test_triplets.txt', augmenting=False)
test_data = test_data.batch(TEST_BATCH_SIZE).prefetch(2)
prediction_model = create_prediction_model(model)
predictions = prediction_model.predict(test_data, steps=int(np.ceil(num_test_data / TEST_BATCH_SIZE)))

# save prediction to file
np.savetxt('predictions.txt', predictions, fmt='%i')
print("Finished!")
