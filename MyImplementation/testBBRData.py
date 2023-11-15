"""
Author: Troy Daniels

This file was created to check the data. The first method is to see the general sizing and format of the images. 
The second is to visualize the image with its corresponding bbox
"""

import cv2 as cv

# ofile = open("./archive/bbox.csv")
# ofile.readline()
# max_height = 0
# max_width = 0
# max_size = 0
# min_height = None
# min_width = None
# min_size = None
# ave_height = 0
# ave_width = 0
# ave_size = 0
# count = 0
# for line in ofile:
#     bboxdata =line.split(",")
#     x1 = int(bboxdata[1])
#     y1 = int(bboxdata[2])
#     x2 = int(bboxdata[3])
#     y2 = int(bboxdata[4])
#     imgName = bboxdata[0]
#     imgPath = "./archive/images/" + imgName
#     img = cv.imread(imgPath)
#     h,w,c = img.shape
#     size = h * w

#     ave_height+=h
#     ave_width+=w
#     ave_size+= size
#     count+=1

#     if h>max_height:
#         max_height = h
#     if min_height == None or h<min_height:
#         min_height = h

#     if w>max_width:
#         max_width = w
#     if min_width == None or w<min_width:
#         min_width = w

#     if size>max_size:
#         max_size = size
#     if min_size == None or size<min_size:
#         min_size = size
# ave_height /= count
# ave_width /= count
# ave_size /= count
# print(f'the max height is {max_height}\n the min height is {min_height}\n the ave height is {ave_height}\n the max width is {max_width}\n the min width is {min_width}\n the average width is {ave_width}\n the max size is {max_size}\n the min size is {min_size}\n the ave size is {ave_size}')
"""
The answer I got above is - 
the max height is 4752
 the min height is 1049
 the ave height is 2716.3612750885477
 the max width is 5184
 the min width is 1440
 the average width is 2571.219598583235
 the max size is 17915904
 the min size is 2073600
 the ave size is 6991014.002361275

this is to see the data, see the image with the bounding box (label drawn)
"""

# ofile = open("./archive/bbox.csv")
# ofile.readline()
# for line in ofile:
#     bboxdata =line.split(",")
#     x1 = int(bboxdata[1])
#     y1 = int(bboxdata[2])
#     x2 = int(bboxdata[3])
#     y2 = int(bboxdata[4])
#     imgName = bboxdata[0]
#     imgPath = "./archive/images/" + imgName
#     image = cv.imread(imgPath)
#     cv.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 5)
#     cv.imshow('test', image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
    

# image = cv2.imread(image, cv2.IMREAD_COLOR)

# cv2.putText(image, f"IOU: {iou:.4f}", (x,y), cv2.FONT_HERSHEY_SIMPLEX,font_size, (255,0,0),3)
#     cv2.imwrite(f"resultsImage", image)

"""

This is to see a visual representation of the sliding window approch

"""

ofile = open("./archive/bbox.csv")
ofile.readline()
for line in ofile:
    bboxdata =line.split(",")
    imgName = bboxdata[0]
    imgPath = "./archive/images/" + imgName
    image = cv.imread(imgPath)
    image = cv.resize(image, (500,500))
    for i in range(0,500,200):
        for j in range(0,500,200):
            cv.rectangle(image, (j,i), (j+200,i+200), (i,j,i+j), 2)
    cv.imshow('test', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    