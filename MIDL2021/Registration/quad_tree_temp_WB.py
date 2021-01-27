#https://scipython.com/blog/quadtrees-2-implementation-in-python/
#https://pydoc.net/openslide-python/1.1.1/openslide/

import json
import pickle
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

if __name__ == '__main__':

    folder = Path("MIDL2021/Registration/")
    annotations = pd.read_csv(folder / "Validation/GT.csv")

    slide_folder = Path("D:/Datasets/ScannerStudy")
    if slide_folder.exists() == False:
        slide_folder = Path("/data/ScannerStudy")
    if slide_folder.exists() == False:
        slide_folder = Path("/mnt/d/Datasets/ScannerStudy")
    if slide_folder.exists() == False:
        slide_folder = Path("/data/ScannerStudy")

    slide_files = {path.name: path for path in slide_folder.glob("*/*/*.*")}

    image_type = "CCMCT"# CCMCT

    annotations = annotations[annotations["image_type"] == image_type]

    annotations["image_name_stem"] = [Path(image_name).stem for image_name in annotations["image_name"]]
    annotations["patient_id"] = [name.split("_")[2] for name in annotations["image_name"]]

    annotations["x1"] = [json.loads(vector.replace("\'","\""))['x1'] for vector in annotations["vector"]]
    annotations["y1"] = [json.loads(vector.replace("\'","\""))['y1'] for vector in annotations["vector"]]

    annotations["x2"] = [json.loads(vector.replace("\'","\""))['x2'] for vector in annotations["vector"]]
    annotations["y2"] = [json.loads(vector.replace("\'","\""))['y2'] for vector in annotations["vector"]]

    annotations["center_x"] = [x1 + ((x2-x1) / 2) for x1, x2 in zip(annotations["x1"], annotations["x2"])]
    annotations["center_y"] = [y1 + ((y2-y1) / 2) for y1, y2 in zip(annotations["y1"], annotations["y2"])]

    annotations["center"] = [np.array((center_x, center_y)) for center_x, center_y in zip(annotations["center_x"], annotations["center_y"])]

    annotations["anno_width"] = [x2-x1 for x1, x2 in zip(annotations["x1"], annotations["x2"])]
    annotations["anno_height"]= [y2-y1 for y1, y2 in zip(annotations["y1"], annotations["y2"])]

    source_scanner_annotations = annotations[annotations["scanner"] == "Aperio"]
    
    for patient_id in source_scanner_annotations["patient_id"].unique():

        source_annos = source_scanner_annotations[source_scanner_annotations["patient_id"] == patient_id]
        source_anno = source_annos.iloc[0]

        target_patient_annotations = annotations[annotations["patient_id"] == patient_id]

        for target_image_name in target_patient_annotations["image_name"].unique():

            target_annos = target_patient_annotations[target_patient_annotations["image_name"] == target_image_name]
            target_anno = target_annos.iloc[0]

            if source_anno.scanner == taret_anno.scanner:
                continue

            source_slide = openslide.OpenSlide(str(slide_files[source_anno.image_name]))
            target_slide = openslide.OpenSlide(str(slide_files[target_anno.image_name]))

            source_dimension = Rect.create(Rect, 0, 0, source_slide.dimensions[0], source_slide.dimensions[1])
            target_dimension = Rect.create(Rect, 0, 0, target_slide.dimensions[0], target_slide.dimensions[1])

            parameter = {
                # feature extractor parameters
                "point_extractor": "sift",  #orb , sift
                "maxFeatures": 512, 
                "crossCheck": False, 
                "flann": False,
                "ratio": 0.5, 
                "use_gray": True,

                # QTree parameter 
                "homography": False,
                "filter_outliner": False,
                "debug": True,
                "target_depth": 1,
                "run_async": False,
                "thumbnail_size": (1024, 1024)
            }


            qtree = QuadTree(source_dimension, source_slide, target_dimension, target_slide, **parameter)


            dist_list = []

            intersections = list(set(source_annos["type_name"]).intersection(target_annos["type_name"]))

            for type_name in intersections:

                source_anno = source_annos[source_annos["type_name"] == type_name].iloc[0]
                target_anno = target_annos[target_annos["type_name"] == type_name].iloc[0]

                box = [source_anno.center_x, source_anno.center_y, source_anno.anno_width, source_anno.anno_height]
                target_box = [target_anno.center_x, target_anno.center_y, target_anno.anno_width, target_anno.anno_height]

                trans_box = qtree.transform_boxes(np.array([box]))[0]

                distance = np.linalg.norm(target_box[:2]-trans_box[:2])

                dist_list.append(distance)
            
            print(f"{target_image_name} {np.array(dist_list).mean()}")


