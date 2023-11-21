import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import ModelFunctions
import testBBRData


(training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels) = ModelFunctions.splitImages(skip_data=760)



def seeObjectivenessPrediction(testing_data, testing_labels, model_name = "objectivness_score_model.hs"):
    model = keras.models.load_model(model_name)
    print(testing_data.size)
    y_pred = model.predict(testing_data)
    print(y_pred)
    for i in range(len(testing_data)):
        print("Predicted value is: "+str(y_pred[i])+" and true value is: "+str(testing_labels[i]))
        cv.imshow('test', testing_data[i])
        cv.waitKey(0)        
        cv.destroyAllWindows()


seeObjectivenessPrediction(testing_data,testing_labels)