# %%
import random
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from numpy.random import seed
from tensorflow.keras import Input, Sequential, callbacks, layers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.regularizers import L2
from tensorflow.random import set_seed
from wandb.keras import WandbCallback
import wandb
from helpers import *

IS_SUBMISSION = True
IMAGE_SIZE = 28
CHANNELS = 1
# %%
# Import dataset
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
data = train.append(test)
# %%
# Reformat data
[X, y] = split_predictor(train, "label")
# %%
# Initialize run
wandb.login(key="de2bb0d0415ffbdb950cf515e5fabff4fb8350b6")
run = wandb.init(
    project="digits",
    tags=["kaggle"] if IS_SUBMISSION else [],
    magic=not IS_SUBMISSION,
    config={
        "dataset": "MNIST",
        "layer_1_filters": 32,
        "layer_1_kernel_size": 3,
        "layer_2_filters": 64,
        "layer_2_kernel_size": 3,
        "layer_dense_1_units": 128,
        "kernel_regularization": 0.0005,
        "dropout": 0.25,
        "optimizer": "nadam",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epochs": 200,
        "batch_size": 64,
        "early_stopping_min_delta": 0.1,
        "early_stopping_patience": 10,
        "augmentation_ratio": 0.1,
        "validation_split": 0.05,
        "random_seed": 42,
    },
)

config = wandb.config
seed(config.random_seed)
set_seed(config.random_seed)
random.seed(config.random_seed)
# %%
# Compile model
def dropout_layer():
    return Dropout(config.dropout, seed=config.random_seed)


def convolutional_layer(filters, kernel_size):
    return [
        Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_regularizer=L2(config.kernel_regularization),
            activation="relu",
        ),
        Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            use_bias=False,
        ),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(),
        dropout_layer(),
    ]


def dense_layer(units):
    return [
        Dense(units=units, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
    ]


model = Sequential(
    [
        # Augment data
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
        # Extract small features
        *convolutional_layer(config.layer_1_filters, config.layer_1_kernel_size),
        # Extract medium features
        *convolutional_layer(config.layer_2_filters, config.layer_2_kernel_size),
        # Flatten to 1D
        Flatten(),
        # Classify
        *dense_layer(config.layer_dense_1_units * 2),
        *dense_layer(config.layer_dense_1_units),
        *dense_layer(config.layer_dense_1_units / 2),
        dropout_layer(),
        Dense(units=10, activation="softmax"),
    ]
)

model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])
# %%
# Define callbacks
learning_rate = callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=2, verbose=1, factor=0.2
)

early_stopping = callbacks.EarlyStopping(
    min_delta=config.early_stopping_min_delta,
    patience=config.early_stopping_patience,
    monitor="val_accuracy",
    restore_best_weights=True,
)
# %%
# Fit data to model
X = Reshape((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))(X)
X = layers.experimental.preprocessing.Rescaling(1 / 255)(X)

test = Reshape((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))(test)
test = layers.experimental.preprocessing.Rescaling(1 / 255)(test)

data_generator = ImageDataGenerator(
    validation_split=config.validation_split,
    height_shift_range=config.augmentation_ratio,
    width_shift_range=config.augmentation_ratio,
    zoom_range=config.augmentation_ratio,
    rotation_range=config.augmentation_ratio * 100,
    shear_range=config.augmentation_ratio,
)

train_generator = data_generator.flow(
    X, y, batch_size=config.batch_size, seed=config.random_seed, subset="training"
)

validation_generator = data_generator.flow(
    X, y, batch_size=config.batch_size, seed=config.random_seed, subset="validation"
)

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config.epochs,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=[learning_rate, early_stopping, WandbCallback()],
    verbose=1,
)
# %%
# Run predictions
predictions = model.predict(test)
submissions = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submissions.Label = np.argmax(predictions, axis=1)
submissions.to_csv("submissions.csv", index=False)

# %%
# Calculate test accuracy
from keras.datasets import mnist

(full_X_train, full_y_train), (full_X_test, full_y_test) = mnist.load_data()
full_data = pd.DataFrame(np.concatenate((full_X_train, full_X_test)).reshape(-1, 784))
full_data.columns = ["pixel" + str(column) for column in full_data.columns]
full_data["label"] = np.concatenate((full_y_train, full_y_test))

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")
cols = test.columns
test["dataset"] = "test"
train["dataset"] = "train"
dataset = pd.concat([train.drop("label", axis=1), test]).reset_index()

labels = full_data["label"].values
full_data.drop("label", axis=1, inplace=True)
full_data.columns = cols

idx_mnist = full_data.sort_values(by=list(full_data.columns)).index
dataset_from = dataset.sort_values(by=list(full_data.columns))["dataset"].values
original_idx = dataset.sort_values(by=list(full_data.columns))["index"].values

for i in range(len(idx_mnist)):
    if dataset_from[i] == "test":
        sample_submission.loc[original_idx[i], "Label"] = labels[idx_mnist[i]]

submissions["RealLabel"] = sample_submission["Label"]
correct = submissions[submissions.Label == submissions.RealLabel]

wandb.log({"test_accuracy": len(correct) / len(test)})
# %%
# Save metrics
wandb.finish()
# %%
