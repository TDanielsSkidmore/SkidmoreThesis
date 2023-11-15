import tensorflow as tf
import ModelFunctions
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def build_model(input_shape):
    inputs= L.Input(input_shape)
    print(inputs.shape)


    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        alpha=0.75
    )
    # backbone.summary()


    x = backbone.output
    # x = backbone.get_layer("block_13_expand_relu").output
    x = L.Conv2D(256,kernel_size=1,padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(4)(x)
    # print(x.shape)

    #add the output layer to have 5 in which the 5th output would be how close the predicted bounding box has the object located in the center

    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    input_shape = (256,256,3)
    model = build_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanSquaredError())
    model.summary()
    (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.load_data()
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("MobileNet_bounding_box_model.hs", save_best_only = True)
    history = model.fit(training_data, training_labels, epochs=500, validation_data= (validation_data, validation_labels), callbacks = [checkpoint_cb])
    # model = tf.keras.model.load_model("my_keras_model.hs")

    # history = model.fit(training_data, training_labels, epochs=250, validation_data= (validation_data, validation_labels))


    # tf.keras.utils.plot_model(model)