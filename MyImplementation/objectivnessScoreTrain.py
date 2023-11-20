"""
Author: Troy Daniels
"""
import ModelFunctions
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

(training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.splitImages()

model = keras.Sequential([
    tf.keras.layers.Conv2D(24, kernel_size=(3,3), padding='same',activation='relu',
                           input_shape=(256,256,3)),
    tf.keras.layers.MaxPooling2D(pool_size=(3,3)),
    tf.keras.layers.Conv2D(40, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(42, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(70, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(50, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(90, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(70, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(100, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(80, kernel_size=(3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
model.summary()
# tf.keras.utils.plot_model(model,to_file="my_model.png")

checkpoint_cb = keras.callbacks.ModelCheckpoint("objectivness_score_model.hs", save_best_only = True)
history = model.fit(training_data, training_labels, epochs=500, validation_data= (validation_data, validation_labels),callbacks = [checkpoint_cb])
pd.DataFrame(history.history).plot()
plt.show()
plt.savefig("objectivness_score_loss")
