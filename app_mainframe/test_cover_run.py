from __future__ import print_function
from opencv_webapp.views import CoverDescriptor
from opencv_webapp.views import CoverMatcher
import glob
import csv
import cv2

# initialize settings
db_path = 'aliens_db.csv'
covers_path = 'aliens'
query_path = 'queries/Root_Leinwand.jpg'

# initialize the database dictionary of covers
db = {}

# loop over the database
for l in csv.reader(open(db_path)):
    # update the database using the image ID as the key
    db[l[0]] = l[1:]

# initialize the default parameters using BRISK is being used
useSIFT = False
useHamming = True
ratio = 0.7
minMatches = 40

# if SIFT is to be used, then update the parameters
if useSIFT:
    minMatches = 50

# initialize the cover descriptor and cover matcher
cd = CoverDescriptor(useSIFT=useSIFT)
cv = CoverMatcher(cd, glob.glob(covers_path + "/*.jpg"),
                  ratio=ratio, minMatches=minMatches, useHamming=useHamming)

# load the query image, convert it to grayscale, and extract
# keypoints and descriptors
queryImage = cv2.imread(query_path)
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)

# try to match the book cover to a known database of images
results = cv.search(queryKps, queryDescs)

# show the query cover
cv2.imshow("Query", queryImage)

# check to see if no results were found
if len(results) == 0:
    print("I could not find a match for that cover!")
    cv2.waitKey(0)

# otherwise, matches were found
else:
    print(results)
    # loop over the results
    for (i, (score, coverPath)) in enumerate(results):
        # grab the book information

        (author, title) = db[coverPath[coverPath.rfind("\\") + 1:]]
        print("{}. {:.2f}% : {} - {}".format(i + 1, score * 100,
                                             author, title))

        # load the result image and show it
        result = cv2.imread(coverPath)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
