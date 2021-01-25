#https://scipython.com/blog/quadtrees-2-implementation-in-python/
#https://pydoc.net/openslide-python/1.1.1/openslide/

import pickle
import json
import numpy as np
import openslide
from probreg import cpd
import cv2
import pandas as pd
from PIL import Image
import imutils
from pathlib import Path
from probreg import transformation as tf
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib as mpl

from registration_tree import Rect, QuadTree

folder = Path("MIDL2021/Registration/")
annotations = pd.read_csv(folder / "Validation/GT.csv")

annotations["image_name_stem"] = [Path(image_name).stem for image_name in annotations["image_name"]]

annotations["x1"] = [json.loads(vector.replace("\'","\""))['x1'] for vector in annotations["vector"]]
annotations["y1"] = [json.loads(vector.replace("\'","\""))['y1'] for vector in annotations["vector"]]

annotations["x2"] = [json.loads(vector.replace("\'","\""))['x2'] for vector in annotations["vector"]]
annotations["y2"] = [json.loads(vector.replace("\'","\""))['y2'] for vector in annotations["vector"]]

annotations["center_x"] = [x1 + ((x2-x1) / 2) for x1, x2 in zip(annotations["x1"], annotations["x2"])]
annotations["center_y"] = [y1 + ((y2-y1) / 2) for y1, y2 in zip(annotations["y1"], annotations["y2"])]

annotations["anno_width"] = [x2-x1 for x1, x2 in zip(annotations["x1"], annotations["x2"])]
annotations["anno_height"]= [y2-y1 for y1, y2 in zip(annotations["y1"], annotations["y2"])]

image_type = "CCMCT"
source_image_name, source_scanner = Path("A_CCMCT_183715A_1.svs"), "Aperio"
target_image_name, target_scanner = Path("N2_CCMCT_183715A_1.ndpi"), "NanoZoomer2.0HT"  #"NanoZoomerS210"

#source_slide = openslide.OpenSlide(f'D:/Datasets/ScannerStudy/{source_scanner}/{image_type}/{source_image_name.name}')
#target_slide = openslide.OpenSlide(f'D:/Datasets/ScannerStudy/{target_scanner}/{image_type}/{target_image_name.name}')

qtree = pickle.load(open(str(folder / f"{image_type}/Depth_2/{source_image_name.stem}-To-{target_image_name.stem}.pickle"), "rb" )) 

source_annos = annotations[annotations["image_name_stem"] == source_image_name.stem]
target_annos = annotations[annotations["image_name_stem"] == target_image_name.stem]


temp_boxes = []

for type_name in source_annos["type_name"].unique():

    source_anno = source_annos[source_annos["type_name"] == type_name].iloc[0]
    target_anno = target_annos[target_annos["type_name"] == type_name].iloc[0]

    box = [source_anno.center_x, source_anno.center_y, source_anno.anno_width, source_anno.anno_height]
    target_box = [target_anno.center_x, target_anno.center_y, target_anno.anno_width, target_anno.anno_height]

    trans_box = qtree.transform_boxes(np.array([box]))


trans_box = qtree.transform_boxes(temp_boxes)
print("hallo")




