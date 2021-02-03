# %%
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, callbacks
from wandb.keras import WandbCallback
from helpers import *

# %%
data = pd.read_csv("../input/digit-recognizer/train.csv")

wandb.init(
    project="digits",
    config={
        "optimizer": "adam",
        "loss": "mae",
        "metric": "accuracy",
        "epochs": 200,
        "batch_size": 32,
        "units": 256,
        "early_stopping_min_delta": 0.01,
        "early_stopping_patience": 10,
    },
)

config = wandb.config
# %%
[X, y] = split_predictor(data, "label")

X_train, X_test, y_train, y_test = train_test_split(X, y)

# RESHAAAAPPPE
# %%
model = Sequential(
    [
        layers.Dense(config.units, input_shape=[784]),
        layers.Dense(1),
    ]
)

model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])

early_stopping = callbacks.EarlyStopping(
    monitor="accuracy",
    min_delta=config.early_stopping_min_delta,
    patience=config.early_stopping_patience,
    restore_best_weights=True,
)

# Fit data to model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=[early_stopping, WandbCallback()],
    verbose=0,
)

wandb.finish()
# %%