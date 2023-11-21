import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import ModelFunctions
import testBBRData


(training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.splitImages(skip_data=760)

one_data = training_data[0]
cv.imshow('test', one_data)
cv.waitKey(0)        
cv.destroyAllWindows()
model = keras.models.load_model("objectivness_score_model.hs")
y_pred = model.predict(one_data)


def seeObjectivenessPrediction(testing_data, testing_labels, model_name = "objectivness_score_model.hs"):
    model = keras.models.load_model(model_name)
    print(testing_data.size)
    y_pred = model.predict(testing_data)
    print(y_pred)
    for i in range(len(testing_data)):
        print("Predicted value is: "+str(y_pred[i])+" and true value is: "+str(testing_labels[i]))