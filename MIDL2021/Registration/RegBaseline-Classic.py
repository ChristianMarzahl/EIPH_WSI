import numpy as np
import pandas as pd
from pathlib import Path
import openslide
import matplotlib.pyplot as plt
import cv2
import imutils
from enum import Enum

class RegMethod(Enum):
    Homography = 0,
    Affine = 1,
    
    
    
def align_images(source_image, target_image, maxFeatures:int=500, 
                 keepPercent:float=0.2, debug=False, 
                 source_scale:tuple=(1,1), target_scale:tuple=(1,1)):

    source_scale = np.array(source_scale)
    target_scale = np.array(target_scale)
    
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
    
    
    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)
    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(source_image, kpsA, target_image, kpsB,matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(matchedVis)
        plt.tight_layout()
        
        
    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt * source_scale 
        ptsB[i] = kpsB[m.trainIdx].pt * target_scale
        
    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    return H
        
        
file_scanner = {
    "24_EIPH_576255 Berliner Blau.svs": {"Name": "Aperio", "slide": None, "thumbnail": None},
    "Z_BB_576255_1.tif": {"Name":"AxioScan", "slide": None, "thumbnail": None},
    "N2_BB_576255_1.ndpi": {"Name": "NanoZoomer2.0HT", "slide": None, "thumbnail": None},
    "N1_BB_576255_1.ndpi": {"Name": "NanoZoomerS210", "slide": None, "thumbnail": None},
}


df = pd.read_pickle("MIDL2021/Registration/reg_results.p")

df["center_x"] = [x1 + ((x2-x1) / 2) for x1, x2 in zip(df["x1"], df["x2"])]
df["center_y"] = [y1 + ((y2-y1) / 2) for y1, y2 in zip(df["y1"], df["y2"])]

df["scanner"] = [file_scanner[file]["Name"] for file in df["file"]]


for id, path in enumerate(df["file"].unique()):
    
    slide = openslide.OpenSlide(filename=f'MIDL2021/Slides/{file_scanner[path]["Name"]}/{path}')
    file_scanner[path]["slide"] = slide
    
    thumbnail = slide.get_thumbnail(size=(2048, 2048))
    height, width = thumbnail.height, thumbnail.width
    
    scale_y, scale_x = np.array(slide.dimensions) / np.array([thumbnail.height, thumbnail.width])
    file_scanner[path]["scale"] = np.array([scale_x, scale_y])

    file_scanner[path]["thumbnail"] = thumbnail



source_file = "24_EIPH_576255 Berliner Blau.svs"
source_image = np.array(file_scanner[source_file]["thumbnail"])
source_scale = file_scanner[source_file]["scale"]

for path in list(df["file"].unique())[1:]:
    target_image = np.array(file_scanner[path]["thumbnail"])
    target_scale = file_scanner[path]["scale"]
    
    H = align_images(source_image, target_image, source_scale=source_scale, target_scale=target_scale, debug=True)

    file_scanner[path]["Homography"] = H