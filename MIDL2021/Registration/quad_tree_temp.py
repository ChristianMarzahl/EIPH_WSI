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
from pathlib import Path
from probreg import transformation as tf
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib as mpl

from registration_tree import Rect, QuadTree

if __name__ == '__main__':

    folder = Path("MIDL2021/Registration/")
    annotations = pd.read_csv(folder / "Validation/GT.csv")

    annotations["image_name_stem"] = [Path(image_name).stem for image_name in annotations["image_name"]]

    annotations["x1"] = [json.loads(vector.replace("\'","\""))['x1'] for vector in annotations["vector"]]
    annotations["y1"] = [json.loads(vector.replace("\'","\""))['y1'] for vector in annotations["vector"]]

    annotations["x2"] = [json.loads(vector.replace("\'","\""))['x2'] for vector in annotations["vector"]]
    annotations["y2"] = [json.loads(vector.replace("\'","\""))['y2'] for vector in annotations["vector"]]

    annotations["center_x"] = [x1 + ((x2-x1) / 2) for x1, x2 in zip(annotations["x1"], annotations["x2"])]
    annotations["center_y"] = [y1 + ((y2-y1) / 2) for y1, y2 in zip(annotations["y1"], annotations["y2"])]

    annotations["center"] = [np.array((center_x, center_y)) for center_x, center_y in zip(annotations["center_x"], annotations["center_y"])]

    annotations["anno_width"] = [x2-x1 for x1, x2 in zip(annotations["x1"], annotations["x2"])]
    annotations["anno_height"]= [y2-y1 for y1, y2 in zip(annotations["y1"], annotations["y2"])]


    source_image_name = Path("A_CCMCT_183715A_1.svs")
    target_image_name = Path("N1_CCMCT_183715A_1.ndpi")

    source_slide = openslide.OpenSlide(f'D:/Datasets/ScannerStudy/Aperio/CCMCT/{str(source_image_name)}')
    target_slide = openslide.OpenSlide(f'D:/Datasets/ScannerStudy/NanoZoomerS210/CCMCT/{str(target_image_name)}')

    #source_slide = openslide.OpenSlide(f'/data/ScannerStudy/Aperio/CCMCT/{str(source_image_name)}')
    #target_slide = openslide.OpenSlide(f'/data/ScannerStudy/NanoZoomerS210/CCMCT/{str(target_image_name)}')

    source_dimension = Rect.create(Rect, 0, 0, source_slide.dimensions[0], source_slide.dimensions[1])
    target_dimension = Rect.create(Rect, 0, 0, target_slide.dimensions[0], target_slide.dimensions[1])


    #for thumbnail_size in [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (12288, 12288), (16384, 16384)]:
    for target_depth in [0]: #, 1, 2
        parameter = {
            # feature extractor parameters
            "point_extractor": "sift",  #orb , sift
            "maxFeatures": 512, 
            "crossCheck": False, 
            "flann": False,
            "ratio": 0.5, 
            "use_gray": True,

            # QTree parameter 
            "homography": True,
            "filter_outliner": False,
            "debug": True,
            "target_depth": target_depth,
            "run_async": False,
            "thumbnail_size": (1024, 1024)
        }
        

        qtree = QuadTree(source_dimension, source_slide, target_dimension, target_slide, **parameter)

        print(f"{qtree.run_time}")

        source_annos = annotations[annotations["image_name_stem"] == source_image_name.stem]
        target_annos = annotations[annotations["image_name_stem"] == target_image_name.stem]

        dist_list = []
        for type_name in source_annos["type_name"].unique():

            source_anno = source_annos[source_annos["type_name"] == type_name].iloc[0]
            target_anno = target_annos[target_annos["type_name"] == type_name].iloc[0]

            box = [source_anno.center_x, source_anno.center_y, source_anno.anno_width, source_anno.anno_height]
            target_box = [target_anno.center_x, target_anno.center_y, target_anno.anno_width, target_anno.anno_height]

            trans_box = qtree.transform_boxes(np.array([box]))[0]

            distance = np.linalg.norm(target_box[:2]-trans_box[:2])

            dist_list.append(distance)
            
        print(f"Tub: {target_depth} {np.array(dist_list).mean()}")
    print("")


