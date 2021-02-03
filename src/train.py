# %%
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, callbacks
from wandb.keras import WandbCallback
from helpers import *

# %%
# Import dataset
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

wandb.login(key="de2bb0d0415ffbdb950cf515e5fabff4fb8350b6")
wandb.init(
    project="digits",
    config={
        "layer_1_filters": 32,
        "layer_1_kernel_size": 6,
        "layer_2_filters": 32,
        "layer_2_kernel_size": 6,
        "optimizer": "nadam",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epochs": 200,
        "batch_size": 32,
        "early_stopping_min_delta": 0.01,
        "early_stopping_patience": 10,
    },
)

config = wandb.config
# %%
# Split predictor
[X, y] = split_predictor(train, "label")

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
X_test = test
# %%
# Reshape data
IMAGE_SIZE = 28

X_train = X_train.values.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
X_valid = X_valid.values.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
X_test = X_test.values.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
# %%
# Compile model
model = Sequential(
    [
        layers.Conv2D(
            filters=config.layer_1_filters,
            kernel_size=config.layer_1_kernel_size,
            input_shape=[IMAGE_SIZE, IMAGE_SIZE, 1],
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(
            filters=config.layer_2_filters, kernel_size=config.layer_2_kernel_size
        ),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(units=128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(units=10, activation="softmax"),
    ]
)

model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])

early_stopping = callbacks.EarlyStopping(
    monitor="accuracy",
    min_delta=config.early_stopping_min_delta,
    patience=config.early_stopping_patience,
    restore_best_weights=True,
)
# %%
# Fit data to model
model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=[early_stopping, WandbCallback()],
    verbose=1,
)
# %%
# Run predictions
predictions = model.predict(X_test)
submissions = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submissions.Label = predictions
# %%
wandb.finish()
# %%