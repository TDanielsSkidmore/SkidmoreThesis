
"""
Author: Troy Daniels

notes:
Images should be in a list where each image is of type numpy.ndarray

"""
import cv2 as cv
import numpy as np
import math
import random
import tensorflow as tf

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



def splitImages(filter_zero_lables=True, skip_data=0):
     # lists for data and labels
    images = []
    labels = []
    # open the file and start reading the data in there and saving it to variables
    ofile = open("./archive/bbox_small.csv")
    ofile.readline()
    print("Starting to load data")
    if skip_data>0:
        for i in range(skip_data):
            ofile.readline()
    for line in ofile:
        bboxdata =line.split(",")
        x1_bbox = int(bboxdata[1])
        y1_bbox = int(bboxdata[2])
        x2_bbox = int(bboxdata[3])
        y2_bbox = int(bboxdata[4])
        imgName = bboxdata[0]
        imgPath = "./archive/images/" + imgName
        image = cv.imread(imgPath)
        H, W, C = image.shape
        one_sixth_image_height =  math.floor(H/6)
        one_sixth_image_width =  math.floor(W/6)
        segment_dim_height =  math.floor(H/2)
        segment_dim_width = math.floor(W/2)
        for i in range(segment_dim_height, H ,one_sixth_image_height):
            for j in range(segment_dim_width, W, one_sixth_image_width  ) :
                temp_image = image[i-segment_dim_height:i,j-segment_dim_width:j]

                x1 = j - segment_dim_width
                y1 = i - segment_dim_height
                x2 = j
                y2 = i
                
                x1_intersect = max(x1,x1_bbox)
                y1_intersect = max(y1,y1_bbox)
                x2_intersect = min(x2, x2_bbox)
                y2_intersect = min(y2, y2_bbox)

                if (x1_intersect>x2) or (y1_intersect>y2) or (x2_intersect<x1) or (y2_intersect<y1):
                    objectiveness_label = 0
                else:
                    objectiveness_label = ( (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect) ) / ( (x2_bbox-x1_bbox) * (y2_bbox - y1_bbox) )

                temp_image = cv.resize(temp_image, (256,256))
                temp_image = (temp_image - 127.5) / 127.5
                images.append(temp_image)
                labels.append(objectiveness_label)

        # segment_dim_height =  math.floor( (3 *H) / 4)
        # segment_dim_width = math.floor( (3 * W) / 4)
        # for i in range(segment_dim_height, H ,100):
        #     for j in range(segment_dim_width, W, 100) :
        #         temp_image = image[i-segment_dim_height:i,j-segment_dim_width:j]

        #         x1 = j - segment_dim_width
        #         y1 = i - segment_dim_height
        #         x2 = j
        #         y2 = i
                
        #         x1_intersect = max(x1,x1_bbox)
        #         y1_intersect = max(y1,y1_bbox)
        #         x2_intersect = min(x2, x2_bbox)
        #         y2_intersect = min(y2, y2_bbox)

        #         if (x1_intersect>x2) or (y1_intersect>y2) or (x2_intersect<x1) or (y2_intersect<y1):
        #             objectiveness_label = 0
        #         else:
        #             objectiveness_label = ( (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect) ) / ( (x2_bbox-x1_bbox) * (y2_bbox - y1_bbox) )

        #         temp_image = cv.resize(temp_image, (256,256))
        #         temp_image = (temp_image - 127.5) / 127.5
        #         images.append(temp_image)
        #         labels.append(objectiveness_label)
    if filter_zero_lables:
        images,labels = filter_out_zero_labels(images,labels)
    shuffleData(images,labels)
    training_data = np.array(images[:round(len(images)*0.8)])
    training_labels = np.array(labels[:round(len(images)*0.8)])
    testing_data = np.array(images[round(len(images)*0.8):round(len(images)*0.9)])
    testing_labels = np.array(labels[round(len(images)*0.8):round(len(images)*0.9)])
    validation_data = np.array(images[round(len(images)*0.9):])
    validation_labels = np.array(labels[round(len(images)*0.9):])
    print("Finished loading data")
    return (training_data, training_labels), (testing_data, testing_labels), (validation_data, validation_labels)


def filter_out_zero_labels(images,labels):
    non_zero_count = 0
    zero_count = 0
    for label in labels:
        if label!=0:
            non_zero_count+=1
        else:
            zero_count+=1
    while non_zero_count <= zero_count: 
        random_index = random.randint(0,len(labels))
        if labels[random_index] == 0:
            labels.pop(random_index)
            images.pop(random_index)
            zero_count -= 1
    return images,labels

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

def shuffleData(list_data,list_labels):
    for i in range(5*len(list_data)):
        rand1 = random.randint(0,len(list_data)-1)
        rand2 = random.randint(0,len(list_data)-1)
        randvalue = list_data[rand1]
        list_data[rand1] = list_data[rand2]
        list_data[rand2] = randvalue
        randlabel = list_labels[rand1]
        list_labels[rand1] = list_labels[rand2]
        list_labels[rand2] = randlabel