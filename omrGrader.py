# import these necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#Take input 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

no_of_options = 5
#(Question No. : Answer No.)
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
# load the image
image = cv2.imread(args["image"])
image1 = cv2.imread(args["image"])
#Grey Image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image and create
# a list of contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []
# loop on the contours
for c in cnts:
	# Dimensions of the contours
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)
		
# sort the question contours 
questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]
correct = 0

# loop ove the questions
for (q, i) in enumerate(np.arange(0, len(questionCnts), no_of_options)):
	# sort the contours for the current question from
	# left to right, then initialize the index of the
	# bubbled answer
	cnts = contours.sort_contours(questionCnts[i:i + no_of_options])[0]
	bubbled = None
	
	# loop over the sorted contours
	for (j, c) in enumerate(cnts):
		# construct a mask that reveals only the current "bubble" for the question
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		# apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area 
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)

		# if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)
	# initialize the contour color and the index of the
	# "correct" answer
	color = (0, 0, 255)

        #Question has more number of options
	if q>no_of_options:
                print("Number of options do not match. Error!")
                exit()
                
	k = ANSWER_KEY[q]
	# check to see if the bubbled answer is correct
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1
	# draw the outline of the correct answer on the test
	cv2.drawContours(image, [cnts[k]], -1, color, 3)
# grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
#cv2.putText(image, "{:.2f}%".format(score), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image1)
cv2.imshow("Exam", image)
cv2.waitKey(0)


