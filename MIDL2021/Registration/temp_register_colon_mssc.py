from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


# Python < 3.8?
import sys
#! pip install pickle5

if sys.version_info.minor < 8:
    import pickle5 as pickle

folder = Path("MIDL2021/Registration/")

annotations = pd.read_csv( str(folder / Path("Validation/HE_IHC/GT_D240.csv")))

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

annotations.head()


#for qt, qt_b in [(Path("ColonCancer/Depth_2/2048/CRC-A1-10-To-CRC-A1-10 HE.pickle"), Path("ColonCancer/Depth_0/2048/CRC-A1-10-To-CRC-A1-10 HE.pickle"))]:
for path in (folder / Path("ColonCancer/")).glob("*/*/*-To-*.pickle"):

    qtree = pickle.load(open(str(path), "rb" ))


    source_name, target_name = path.stem.split("-To-")

    patient_id = source_name.split("-")[2].replace("40X", "X40").split(" X40")[0].replace(" HE", "").replace(".tif", "")
    image_type = "MSSC" if "40" not in source_name else "RSC"


    source_annos = annotations[(annotations["image_name_stem"] == source_name) & 
                                   (annotations["image_type"] == image_type)]
    
    if len(source_annos) == 0:
        continue
    
    target_annos = annotations[(annotations["image_name_stem"] == target_name) & 
                                   (annotations["image_type"] == image_type)]
    
    if len(target_annos) == 0:
        continue

    for id, source_anno in source_annos.iterrows():

        if source_anno.type_name != "L23":
            continue
            
        if target_name != "CRC-A1-10 HE":
            continue
            
        if qtree.thumbnail_size[0] != 2048: 
            continue


        print(f"{source_anno.type_name}")
        target_anno = target_annos[target_annos["type_name"] == source_anno.type_name].iloc[0]

        box = [source_anno.center_x, source_anno.center_y, source_anno.anno_width, source_anno.anno_height]

        trans_box = qtree.transform_boxes(np.array([box]))[0]
        distance = 99999

        for i in range(0, qtree.target_depth+1):

            temp_trans_box = qtree.transform_boxes(np.array([box]), i)[0]
            
            temp_distance = np.linalg.norm(target_anno.center-temp_trans_box[:2])
            print(f"{i}:  {temp_trans_box[:2]} ->  {target_anno.center} = {temp_distance}")

            temp_trans_box_0 = qtree.transform_boxes(np.array([box]), i)[0]
            temp_distance_0 = np.linalg.norm(target_anno.center-temp_trans_box_0[:2])

            if temp_distance < distance:
                trans_box = temp_trans_box
                distance = temp_distance 
            



print("")