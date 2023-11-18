"""
Author: Troy Daniels

notes:
Images should be in a list where each image is of type numpy.ndarray

"""
import cv2 as cv
import numpy as np


def load_data():
    # lists for data and labels
    images = []
    bboxes = []
    # open the file and start reading the data in there and saving it to variables
    ofile = open("./archive/bbox.csv")
    ofile.readline()
    print("Starting to load data")
    for line in ofile:
        bboxdata =line.split(",")
        x1 = int(bboxdata[1])
        y1 = int(bboxdata[2])
        x2 = int(bboxdata[3])
        y2 = int(bboxdata[4])
        imgName = bboxdata[0]
        imgPath = "./archive/images/" + imgName
        image = cv.imread(imgPath)
        # get the size of the image. Resize the image. Make the bounding box numbers => {0,1}
        H, W, C = image.shape
        image = cv.resize(image, (256,256))
        image = (image - 127.5)/127.5
        images.append(image)
        norm_x1 = x1 / W
        norm_y1 = y1 / H
        norm_x2 = x2 / W
        norm_y2 = y2 / H
        # add the data/label to the lists
        bbox = [norm_x1,norm_y1,norm_x2,norm_y2]
        bboxes.append(bbox)
    # create the training ,testing and validation
    training_data = np.array(images[:round(len(images)*0.8)])
    training_labels = np.array(bboxes[:round(len(images)*0.8)])
    testing_data = np.array(images[round(len(images)*0.8):round(len(images)*0.9)])
    testing_labels = np.array(bboxes[round(len(images)*0.8):round(len(images)*0.9)])
    validation_data = np.array(images[round(len(images)*0.9):])
    validation_labels = np.array(bboxes[round(len(images)*0.9):])
    print("Finished loading data")
    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)

def IoU(bbox_true, bbox_pred):

    x1 = max(bbox_true[0], bbox_pred[0])
    y1 = max(bbox_true[1], bbox_pred[1])
    x2 = max(bbox_true[2], bbox_pred[2])
    y2 = max(bbox_true[3], bbox_pred[3])

    intersection_area = max(0,x2-x1+1) * max(0,y2-y1+1)

    true_area = (bbox_true[2] - bbox_true[0]+1) * (bbox_true[3] - bbox_true[1] +1)
    bbox_area = (bbox_pred[2] - bbox_pred[0]+1) * (bbox_pred[3] - bbox_pred[1] +1)

    iou = intersection_area / float(true_area + bbox_area - intersection_area)
    return iou