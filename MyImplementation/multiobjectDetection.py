import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import ModelFunctions
import numpy as np
import testBBRData


def multiObjDetectionInImage(objThreshold, iouThreshold, image):

    images ,segementCoordinates = ModelFunctions.segmentImage(image)

    objectivenessModel = keras.models.load_model("MobileNet_objectiveness_model_full_distributed_data.hs")
    boundingBoxModel = keras.models.load_model("MobileNet_bounding_box_model.hs")

    images = np.array(images)

    bboxes = boundingBoxModel.predict(images)
    objectivenessScores = objectivenessModel.predict(images)

    persectiveBboxes = []
    perspectiveObjectivenessScores = []

    finalBboxes = []

    for i in range(len(bboxes)):
        if objectivenessScores[i] > objThreshold:
            persectiveBboxes.append(ModelFunctions.relocateBbox(bboxes[i],segementCoordinates[i]))
            perspectiveObjectivenessScores.append(objectivenessScores[i])
    while len(persectiveBboxes) != 0:
        maxObjectivenessScore = 0
        maxIndex = 0
        for i in range(len(persectiveBboxes)):
            if perspectiveObjectivenessScores[i] > maxObjectivenessScore:
                maxObjectivenessScore = perspectiveObjectivenessScores[i]
                maxIndex = i
        maxBoundingBox = persectiveBboxes.pop(maxIndex)
        perspectiveObjectivenessScores.pop(maxIndex)
        finalBboxes.append(maxBoundingBox)
        temp_list_bboxes = []
        temp_list_objectiveness = []
        for i in range(len(persectiveBboxes)):
            if iouThreshold > ModelFunctions.normalizedOverlap(maxBoundingBox,persectiveBboxes[i]):
                temp_list_bboxes.append(persectiveBboxes[i])
                temp_list_objectiveness.append(perspectiveObjectivenessScores[i])
        perspectiveObjectivenessScores = temp_list_objectiveness
        persectiveBboxes = temp_list_bboxes

    return finalBboxes
        
        
        



def multiObjectDetection(fileName, path = "./multi_person_images/", objThreshold = 0.5, overlapThreshold = 0.4):
    ofile = open(fileName)
    images = []
    bboxSet = []
    line = ofile.readline()
    imageNames = line.split(",")
    for imageName in imageNames:
        imagePath = path+imageName
        image = cv.imread(imagePath)
        images.append(image)
        bboxSet.append(multiObjDetectionInImage(objThreshold,overlapThreshold,image))
    for i in range(len(images)):
        testBBRData.saveBoundingBoxes(images[i],bboxSet[i],"./multiObjectdetectionResults/set2/","image"+str(i))

fileName = "./multi_person_images/set2/multiPersonImages.txt"
bboxes = multiObjectDetection(fileName,path="./multi_person_images/set2/")

