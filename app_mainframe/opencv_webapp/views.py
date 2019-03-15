# from django.shortcuts import render

### Initializing the imports
from __future__ import print_function
import numpy as np
import cv2
import glob
import csv
from django.core import serializers
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


class CoverDescriptor:
    def __init__(self, useSIFT=False):
        # store whether or not SIFT should be used as the feature
        # detector and extractor
        self.useSIFT = useSIFT

    def describe(self, image):
        # initialize the BRISK detector and feature extractor (the
        # standard OpenCV 3 install includes BRISK by default)
        descriptor = cv2.BRISK_create()

        # check if SIFT should be utilized to detect and extract
        # features (this this will cause an error if you are using
        # OpenCV 3.0+ and do not have the `opencv_contrib` module
        # installed and use the `xfeatures2d` package)
        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()

        # detect keypoints in the image, describing the region
        # surrounding each keypoint, then convert the keypoints
        # to a NumPy array
        (kps, descs) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and descriptors
        return (kps, descs)


class CoverMatcher:
    def __init__(self, descriptor, coverPaths, ratio=0.7, minMatches=40,
                 useHamming=True):
        # store the descriptor, book cover paths, ratio and minimum
        # number of matches for the homography calculation, then
        # initialize the distance metric to be used when computing
        # the distance between features
        self.descriptor = descriptor
        self.coverPaths = coverPaths
        self.ratio = ratio
        self.minMatches = minMatches
        self.distanceMethod = "BruteForce"

        # if the Hamming distance should be used, then update the
        # distance method
        if useHamming:
            self.distanceMethod += "-Hamming"

    def search(self, queryKps, queryDescs):
        # initialize the dictionary of results
        results = {}

        # loop over the book cover images
        for coverPath in self.coverPaths:
            # load the query image, convert it to grayscale, and
            # extract keypoints and descriptors
            cover = cv2.imread(coverPath)
            gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
            (kps, descs) = self.descriptor.describe(gray)

            # determine the number of matched, inlier keypoints,
            # then update the results
            score = self.match(queryKps, queryDescs, kps, descs)

            results[coverPath] = score

        # if matches were found, sort them
        if len(results) > 0:
            results = sorted([(v, k) for (k, v) in results.items() if v > 0],
                             reverse=True)

        # return the results
        return results

    def match(self, kpsA, featuresA, kpsB, featuresB):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
        rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # check to see if there are enough matches to process
        if len(matches) > self.minMatches:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])

            # compute the homography between the two sets of points
            # and compute the ratio of matched points
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

            # return the ratio of the number of matched keypoints
            # to the total number of keypoints
            return float(status.sum()) / status.size

        # no matches were found
        return -1.0


def alien_detection(self):
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


@csrf_exempt
def requested_url(request):
    # default value set to be false

    default = {"safely executed": False}  # because no detection yet

    ## between GET or POST, we go with Post request and check for https

    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            default["error_value"] = "There is no Image Provided"
            return JsonResponse(default)

        else:  # URL is provided by the user
            url_provided = request.POST.get("url", None)

            if url_provided is None:
                default["error_value"] = "There is no URL Provided"

                return JsonResponse(default)

            default = alien_detection2(url_provided)
    return JsonResponse(default)


def alien_detection2(queryPath=None):
    queryPath='./queries/Root_Leinwand.jpg'
    # initialize settings
    db_path = 'aliens_db.csv'
    covers_path = 'aliens'
    query_path = queryPath

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
        default = {"no detection"}  # because no detection yet
        return JsonResponse(default)

    # otherwise, matches were found
    else:
        aliens = {}
        # loop over the results
        for (i, (score, coverPath)) in enumerate(results):
            # grab the book information

            (author, title) = db[coverPath[coverPath.rfind("\\") + 1:]]
            print("{}. {:.2f}% : {} - {}".format(i + 1, score * 100,
                                                 author, title))

            # load the result image and show it
            aliens[title] = score*100
        return aliens

