import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import ModelFunctions
import testBBRData


(training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.splitImages(skip_data=700)


# - objectiveness score - make prediction for the test data (I am skipping most of the data so I can load it locally, all is actually part of validation (within 90% of data))
def seeObjectivenessPrediction(testing_data, testing_labels, model_name = "objectivness_score_model.hs"):
    model = keras.models.load_model(model_name)
    y_pred = model.predict(testing_data)
    print(y_pred)
    for i in range(len(testing_data)):
        print("Predicted value is: "+str(y_pred[i])+" and true value is: "+str(testing_labels[i]))

seeObjectivenessPrediction(training_data,training_labels)